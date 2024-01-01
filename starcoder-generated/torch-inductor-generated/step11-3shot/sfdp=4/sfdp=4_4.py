
class Transformer(nn.Module):
    def __init__(self, n_heads: int, embed_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(2003, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim)
        encoder_layer = TransformerEncoderLayer(embed_dim, n_heads)
        self.transformer_encoder = TransformerEncoder(encoder_layer, 10)
        decoder_layer = TransformerDecoderLayer(embed_dim, n_heads)
        self.transformer_encoder = TransformerEncoder(decoder_layer, 10)
 
    def forward(self, src: Tensor, tgt: Tensor, src_mask: Tensor, tgt_mask: Tensor) -> Tensor:
        out = self.embedding(src) * math.sqrt(self.embed_dim)
        out = self.pos_encoder(out)
        encoder_out = self.transformer_encoder(out, src_mask)

        out2 = self.embedding(tgt) * math.sqrt(self.embed_dim)
        out2 = self.pos_encoder(out2)
        decoder_out = self.transformer_encoder(out2, src_mask, tgt_mask)
        return encoder_out

# Initializing the model
m = Transformer(n_heads=2, embed_dim=512)

# Inputs to the model
src = torch.eye(15, 20)
tgt = torch.eye(15, 20)
src_mask = torch.rand_like(src) > 0.5
tgt_mask = torch.rand_like(tgt) > 0.5
