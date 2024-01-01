
class Model(torch.nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, n_encoder_layers,
                 n_decoder_layers, dropout, normalize_before):
 
        super().__init__()
        self.encoder = TransformerEncoder(TransformerEncoderLayer(d_model, nhead, dim_feedforward, normalize_before, dropout), n_encoder_layers)
        self.decoder = TransformerDecoder(TransformerDecoderLayer(d_model, nhead, dim_feedforward, normalize_before, dropout), n_decoder_layers)
 
        self.tgt_word_emb = nn.Linear(d_model, d_model)
        self.decoder_attn = nn.Linear(d_model, d_model)
 
    def forward(self, src: Tensor, tgt: Tensor, tgt_mask: Tensor, src_key_padding_mask: Tensor,
                tgt_key_padding_mask: Tensor, memory_key_padding_mask: Tensor) -> Tensor:
    
        src = src.permute(1, 0, 2)
        tgt = tgt.permute(1, 0, 2)
        tgt_mask = tgt_mask.permute(0, 1, 3, 2)
        src_key_padding_mask = src_key_padding_mask.bool()
        tgt_key_padding_mask = tgt_key_padding_mask.bool()
        memory_key_padding_mask = memory_key_padding_mask.bool()
        src_embedded = self.src_word_emb(src)
        tgt_embedded = self.tgt_word_emb(tgt)
 
        src_embedded = self.encoder(src_embedded, src_key_padding_mask)
        tgt_embedded = self.decoder(tgt_embedded, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask,
                                    memory_key_padding_mask=memory_key_padding_mask)
 
        v = torch.tanh(self.decoder_attn(tgt_embedded))
        v = self.tgt_word_emb.linear(v)
 
        return v.permute(1, 0, 2)
 
 
def PositionalEncoding(d_model, dropout, max_len=1000):
    pe = torch.zeros(max_len, 1, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0, 0::2] = torch.sin(position * div_term)
    pe[:, 0, 1::2] = torch.cos(position * div_term)
    pe = pe.requires_grad_(False)
 
    return nn.BatchNorm1d(d_model)(pe)

# Inputs to the model
src = torch.randint(1, 8192, (1, 8))
tgt = torch.randint(1, 8192, (1, 10))

src_mask = generate_square_subsequent_mask(src.shape[1]).to(src.device)
attn_mask = generate_square_subsequent_mask(tgt.shape[0]).to(src.device)
memory_mask = attn_mask.permute(1, 2, 0)

src_padding_mask = (src == 0).transpose(0, 1)
tgt_padding_mask = (tgt == 0).transpose(0, 1)
memory_padding_mask = (src == 0)

