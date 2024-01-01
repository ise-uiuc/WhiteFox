
class Layer(torch.nn.Module):
    def __init__(self, dim: int, heads: int, dropout_p: float):
        super(TransformerEncoderLayer, self).__init__()
        self.norm1 = SublayerConnection(dim, dropout_p)
        self.norm2 = SublayerConnection(dim, dropout_p)
        self.attn = MultiHeadAttention(heads, dim, dropout_p)
        self.ff = FeedForward(dim)
 
    def forward(self, x: Tensor, mask: Optional[Tensor] = None):
        x = self.attn(x, x, x, mask)
        x = self.norm1(x, sublayer=self.attn)
        x = self.ff(x)
        x = self.norm2(x, sublayer=self.ff)
 
        return x

class TransformerEncoder(torch.nn.Module):
    def __init__(self, num_layers: int, heads: int, dim: int, src_vocab_size: int,
                 dropout_p: float):
        super(TransformerEncoder, self).__init__()
        self.dropout_p = dropout_p
        self.num_layers = num_layers
 
        self.embed = Embeddings(num_embeddings=src_vocab_size,
                            embedding_dim=dim)
 
        self.pe = PositionalEncoding(0)
        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(dim, heads, dropout_p) for _ in range(num_layers)])
 
    def forward(self, src: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        x = self.embed(src)
        x = self.pe(x)
        for layer in self.layers:
            x = layer(x)
 
        return x

# Initializing the model
m = TransformerEncoder(num_layers=2, heads=2, dim=2, src_vocab_size=100, dropout_p=0.1)

# Inputs to the model
x = torch.tensor([[1,2,3], [78, 0, 3]])
