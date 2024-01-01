
class Model(torch.nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, num_decoder_layers, dropout_p=0.5):
        super().__init__()
        self.multi_head_attn = torch.nn.MultiheadAttention(d_model=d_model,
                                                            nhead=nhead,
                                                            dropout=dropout_p)
        self.pos_encoding = self._positional_encoding(size=(num_encoder_layers * 2 + num_decoder_layers, 1, 1, d_model),
                                                      dropout_p=dropout_p)
        self.dropout_p = dropout_p
 
    def _positional_encoding(self, size, dropout_p=0.5):
        dropout = torch.nn.Dropout(p=dropout_p)
        encoding = torch.zeros(size, requires_grad=False)
        pos = torch.arange(size[0]).reshape(*size).to("cuda")
        denominator = torch.exp(torch.arange(0., self.dim, 2) * -(math.log(10000.0) / self.dim))
        encoding[:, :, 0, 0::2] = torch.sin(torch.div(pos, denominator))
        encoding[:, :, 0, 1::2] = torch.cos(torch.div(pos, denominator))
        encoding = dropout(encoding)
        return encoding
 
    def forward(self, q, k, v, pos_seq_len):
        batch_size, encoding_seq_len, chanel_size = q.shape
        pos_seq_len = pos_seq_len[0].numpy()
        encoding_pos = self.pos_encoding[:2 * pos_seq_len + 1]
        encoding_pos = encoding_pos.unsqueeze(1).unsqueeze(1)
        q_encoder_pos = torch.zeros([pos_seq_len, batch_size, self.d_model], device="cuda")
        q_encoder_pos = torch.cat([q_encoder_pos, encoding_pos], dim=0)
        k_encoder_pos = torch.zeros([pos_seq_len, batch_size, self.d_model], device="cuda")
        k_encoder_pos = torch.cat([k_encoder_pos, encoding_pos[1:]], dim=0)
        v_encoder_pos = encoding_pos
        encoder_pos = (q_encoder_pos, k_encoder_pos, v_encoder_pos)
        encoder_inputs = (q, k, v)
        q_decoder_pos = torch.zeros([pos_seq_len, batch_size, self.d_model], device="cuda")
        k_decoder_pos = torch.cat([encoding_pos[1:], torch.zeros([pos_seq_len, batch_size, self.d_model], device="cuda")], dim=0)
        v_decoder_decoder = torch.zeros([pos_seq_len, batch_size, self.d_model], device="cuda")
        decoder_inputs = (q_decoder_pos, k_decoder_pos, v_decoder_decoder)
        query, key, value = self.multi_head_attn(query=q,
                                                key=k,
                                                value=v,
                                                key_padding_mask=None,
                                                need_weights=False,
                                                attn_mask=None,
                                                pos_encoding=encoder_pos,
                                                enc_hidden_state=None,
                                                mask_attn_weights=False)
        decoder_query, decoder_key, decoder_value = self.multi_head_attn(query=query,
                                                                            key=query,
                                                                            value=value,
                                                                            key_padding_mask=None,
                                                                            need_weights=False,
                                                                            attn_mask=None,
                                                                            pos_encoding=None,
                                                                            enc_hidden_state=None,
                                                                            mask_attn_weights=True)
        return query, key, value, decoder_query, decoder_key, decoder_value
 
    def forward_infer(self, q, k, v, pos_seq_len):
        device = torch.device("cuda")
        encoding_seq_len, batch_size, chanel_size = q.shape
        pos_seq_len = pos_seq_len[0].numpy()
        encoding_pos = self.pos_encoding[:2 * pos_seq_len + 1]
        encoding_pos = encoding_pos.unsqueeze(1).unsqueeze(1)
        q_encoder_pos = torch.zeros([pos_seq_len, batch_size, self.d_model], device="cuda")
        q_encoder_pos = torch.cat([q_encoder_pos, encoding_pos], dim=0)
        k_encoder_pos = torch.zeros([pos_seq_len, batch_size, self.d_model], device="cuda")
        k_encoder_pos = torch.cat([k_encoder_pos, encoding_pos[1:]], dim=0)
        v_encoder_pos = encoding_pos
        encoder_pos = (q_encoder_pos, k_encoder_pos, v_encoder_pos)
        encoder_inputs = (q, k, v)
        q_decoder_pos = torch.zeros([pos_seq_len, batch_size, self.d_model], device="cuda")
        k_decoder_pos = torch.cat([encoding_pos[1:], torch.zeros([pos_seq_len, batch_size, self.d_model], device="cuda")], dim=0)
        v_decoder_decoder = torch.zeros([pos_seq_len, batch_size, self.d_model], device="cuda")
        decoder_inputs = (q_decoder_pos, k_decoder_pos, v_decoder_decoder)
        query, key, value = self.multi_head_attn(query=q,
                                                key=k,
                                                value=v,
                                                need_weights=False,
                                                attn_mask=None,
                                                pos_encoding=encoder_pos,
                                                enc_hidden_state=None,
                                                mask_attn_weights=False)
        all_decoder_inputs = torch.nn.utils.rnn.pad_sequence([q_decoder_pos] * 3, batch_first=True).transpose(1, 2).cuda()
        decoder_attention_mask = (torch.sign(torch.abs(all_decoder_inputs) + 3) + 1) * -1e7
        decoder_key_padding_mask = torch.ones([encoding_seq_len - 1, 1], dtype=torch.bool)
        decoder_key_padding_mask = decoder_key_padding_mask.unsqueeze(1).transpose(0, 1)
        decoder_outputs = torch.nn.TransformerDecoder(self.multi_head_attn, 3, None, dropout=self.dropout_p).forward(decoder_inputs,
                                                                                                                     decoder_attention_mask,
                                                                                                                     decoder_key_padding_mask)
        return query, key, value, decoder_query, decoder_key, decoder_value

# Initializing the model
m = Model(d_model=64, nhead=1, num_encoder_layers=1, num_decoder_layers=3)

# Inputs to the model
query = torch.randn([9, 28, 64], requires_grad=True).cuda()
key = torch.randn([9, 28, 64], requires_grad=True).cuda()
value = torch.randn([9, 28, 64], requires_grad=True).cuda()
pos_seq_len = torch.randint(low=2, high=4, size=(1,), requires_grad=False).long().cuda()

# First forward pass
__output__, __output___, __output___, __output___, __output___, 