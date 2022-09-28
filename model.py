import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class WosBaselineModel(nn.Module):
    def __init__(self, config, args):
        super().__init__()
        self.config = config
        self.processor = args.processor

        extra_model_params = (
            "encoder_layerdrop",
            "decoder_layerdrop",
            "dropout",
            "attention_dropout",
        )
        for p in extra_model_params:
            if getattr(args, p, None):
                assert hasattr(
                    self.config, p
                ), f"model config doesn't have a `{p}` attribute"
                setattr(self.config, p, getattr(args, p))

        self.model = AutoModel.from_config(self.config)

        self.processor = args.processor
        self.teacher_forcing = args.teacher_forcing
        self.parallel_decoding = args.parallel_decoding

        self.slot_meta = self.processor.slot_meta
        self.slot_vocab = [
            self.processor.tokenizer.encode(
                slot.replace("-", " "), add_special_tokens=False
            )
            for slot in self.slot_meta
        ]

        self.encoder_config = self.config
        self.encoder = self.model

        self.decoder = SlotGenerator(
            self.encoder_config.vocab_size,
            self.encoder_config.hidden_size,
            args.decoder_hidden_dim,
            self.encoder_config.hidden_dropout_prob,
            self.slot_meta,
            self.processor.gating2id,
            parallel_decoding=args.parallel_decoding,
        )

        self.decoder.set_slot_idx(self.slot_vocab)
        self.tie_weight()

        self.loss_gen = masked_cross_entropy_for_value
        self.loss_gate = nn.CrossEntropyLoss()

    def tie_weight(self):
        """Share the embedding layer for both encoder and decoder"""
        self.decoder.embed.weight = self.encoder.embeddings.word_embeddings.weight

    def forward(
        self, input_ids, token_type_ids, attention_mask=None, max_len=10, teacher=None
    ):
        outputs_dict = self.encoder(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        encoder_outputs = outputs_dict["last_hidden_state"]
        if "pooler_output" in outputs_dict.keys():
            pooler_output = outputs_dict["pooler_output"]
        else:
            pooler_output = self.encoder_pooler_layer(encoder_outputs)

        all_point_outputs, all_gate_outputs = self.decoder(
            input_ids,
            encoder_outputs,
            pooler_output.unsqueeze(0),
            attention_mask,
            max_len,
            teacher,
        )
        return all_point_outputs, all_gate_outputs

    @staticmethod
    def add_arguments(parser):
        parser.add_argument(
            "--teacher_forcing", default=0.5, type=float, help="teacher_forcing"
        )
        parser.add_argument(
            "--decoder_hidden_dim",
            default=400,
            type=int,
            help="hidden dim for rnn_cell in decoder",
        )
        parser.add_argument(
            "--parallel_decoding",
            action="store_true",
            help="Decode all slot-values in parallel manner.",
        )
        return parser


class SlotGenerator(nn.Module):
    def __init__(
        self,
        vocab_size,
        enc_hidden_size,
        dec_hidden_size,
        dropout,
        slot_meta,
        gating2id,
        pad_idx=0,
        parallel_decoding=False,
    ):
        super(SlotGenerator, self).__init__()
        self.hidden_size = dec_hidden_size
        self.pad_idx = pad_idx
        self.slot_meta = slot_meta
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(
            vocab_size, enc_hidden_size, padding_idx=pad_idx
        )  # shared with encoder

        # adjust dims of encoder-related tensor to decoder hidden dim
        self.proj_slot_emb = nn.Linear(enc_hidden_size, self.hidden_size)
        self.proj_enc_hidden = nn.Linear(enc_hidden_size, self.hidden_size)
        self.proj_dec_hidden = nn.Linear(self.hidden_size, enc_hidden_size)

        self.gru = nn.GRU(
            self.hidden_size, self.hidden_size, 1, dropout=dropout, batch_first=True
        )

        # receive gate info from processor
        self.gating2id = (
            gating2id  # {"none": 0, "dontcare": 1, "ptr": 2, "yes":3, "no": 4}
        )
        self.num_gates = len(self.gating2id.keys())

        self.dropout = nn.Dropout(dropout)
        self.w_gen = nn.Linear(self.hidden_size * 3, 1)
        self.sigmoid = nn.Sigmoid()
        self.w_gate = nn.Linear(self.hidden_size, self.num_gates)

        self.slot_embed_idx = []
        self.parallel_decoding = parallel_decoding

    def set_slot_idx(self, slot_vocab_idx):
        whole = []
        max_length = max(map(len, slot_vocab_idx))
        for idx in slot_vocab_idx:
            if len(idx) < max_length:
                gap = max_length - len(idx)
                idx.extend([self.pad_idx] * gap)
            whole.append(idx)
        self.slot_embed_idx = whole  # torch.LongTensor(whole)

    def forward(
        self, input_ids, encoder_output, hidden, input_masks, max_len, teacher=None
    ):
        input_masks = input_masks.ne(1)
        # J, slot_meta : key : [domain, slot] ex> LongTensor([1,2])
        # J,2
        # proj encoder_output
        encoder_output = self.proj_enc_hidden(encoder_output)
        # proj encoder_hidden
        hidden = self.proj_enc_hidden(hidden)

        batch_size = encoder_output.size(0)
        slot = torch.LongTensor(self.slot_embed_idx).to(input_ids.device)
        # proj slot_embedding
        slot_e = torch.sum(self.proj_slot_emb(self.embed(slot)), 1)  # J, d
        J = slot_e.size(0)

        if self.parallel_decoding:
            all_point_outputs = torch.zeros(batch_size, J, max_len, self.vocab_size).to(
                input_ids.device
            )
            all_gate_outputs = torch.zeros(batch_size, J, self.num_gates).to(
                input_ids.device
            )

            w = slot_e.repeat(batch_size, 1).unsqueeze(1)
            hidden = hidden.repeat_interleave(J, dim=1)
            encoder_output = encoder_output.repeat_interleave(J, dim=0)
            input_ids = input_ids.repeat_interleave(J, dim=0)
            input_masks = input_masks.repeat_interleave(J, dim=0)
            num_decoding = 1

        else:
            # Seperate Decoding
            all_point_outputs = torch.zeros(J, batch_size, max_len, self.vocab_size).to(
                input_ids.device
            )
            all_gate_outputs = torch.zeros(J, batch_size, self.num_gates).to(
                input_ids.device
            )
            num_decoding = J

        for j in range(num_decoding):

            if not self.parallel_decoding:
                w = slot_e[j].expand(batch_size, 1, self.hidden_size)

            for k in range(max_len):
                w = self.dropout(w)
                _, hidden = self.gru(w, hidden)  # 1,B,D

                # B,T,D * B,D,1 => B,T
                attn_e = torch.bmm(encoder_output, hidden.permute(1, 2, 0))  # B,T,1
                attn_e = attn_e.squeeze(-1).masked_fill(input_masks, -1e9)
                attn_history = F.softmax(attn_e, -1)  # B,T

                # B,D * D,V => B,V
                attn_v = torch.matmul(
                    self.proj_dec_hidden(hidden.squeeze(0)),
                    self.embed.weight.transpose(0, 1),
                )  # B,V
                attn_vocab = F.softmax(attn_v, -1)

                # B,1,T * B,T,D => B,1,D
                context = torch.bmm(attn_history.unsqueeze(1), encoder_output)  # B,1,D
                p_gen = self.sigmoid(
                    self.w_gen(torch.cat([w, hidden.transpose(0, 1), context], -1))
                )  # B,1
                p_gen = p_gen.squeeze(-1)

                p_context_ptr = torch.zeros_like(attn_vocab).to(input_ids.device)
                p_context_ptr.scatter_add_(1, input_ids, attn_history)  # copy B,V
                p_final = p_gen * attn_vocab + (1 - p_gen) * p_context_ptr  # B,V
                _, w_idx = p_final.max(-1)

                if teacher is not None:
                    if self.parallel_decoding:
                        w = self.embed(teacher[:, :, k]).reshape(batch_size * J, 1, -1)
                    else:
                        w = self.embed(teacher[:, j, k]).unsqueeze(1)
                else:
                    w = self.embed(w_idx).unsqueeze(1)  # B,1,D
                w = self.proj_slot_emb(w)

                if k == 0:
                    gated_logit = self.w_gate(context.squeeze(1))  # B,3
                    if self.parallel_decoding:
                        all_gate_outputs = gated_logit.view(
                            batch_size, J, self.num_gates
                        )
                    else:
                        _, gated = gated_logit.max(1)  # maybe `-1` would be more clear
                        all_gate_outputs[j] = gated_logit

                if self.parallel_decoding:
                    all_point_outputs[:, :, k, :] = p_final.view(
                        batch_size, J, self.vocab_size
                    )
                else:
                    all_point_outputs[j, :, k, :] = p_final

        if not self.parallel_decoding:
            all_point_outputs = all_point_outputs.transpose(0, 1)
            all_gate_outputs = all_gate_outputs.transpose(0, 1)

        return all_point_outputs, all_gate_outputs


def masked_cross_entropy_for_value(logits, target, pad_idx=0):
    mask = target.ne(pad_idx)
    logits_flat = logits.view(-1, logits.size(-1))
    log_probs_flat = torch.log(logits_flat)
    target_flat = target.view(-1, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    losses = losses_flat.view(*target.size())
    losses = losses * mask.float()
    loss = losses.sum() / (mask.sum().float())
    return loss
