
class DotProductAttention(nn.Module):
    def __init__(self, dim, dropout=0.0):
        super(DotProductAttention, self).__init__()
        self.dropout = dropout
        self.scale_factor = torch.sqrt(torch.FloatTensor([dim]))
        
    def forward(self, q, k, v, mask):
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk * self.scale_factor
        softmax_qk = nn.functional.softmax(scaled_qk, dim=-1)
        dropout_qk = nn.functional.drop(softmax_qk, p=self.dropout, training=self.training)
        attentioned_v = dropout_qk.matmul(v)
        return attentioned_v
 
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads, dropout=0.0):
        super().__init__()
        self.attentions = nn.ModuleList()
        for _ in range(num_heads):
            self.attentions.append(DotProductAttention(embed_size, dropout=dropout))
        
    def forward(self, q, k, v, mask):
        output = [att(q, k, v, mask) for att in self.attentions]
        return torch.cat(output, dim=-2)
 
class Transformer(nn.Module):
    def __init__(self, embedding_size=512, num_heads=6, num_encoder_layers=6, dropout=0.0):
        super(Transformer, self).__init__()
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.embed = nn.Conv2d(in_channels=3, out_channels=embedding_size, kernel_size=1)
        self.pos_embed = PositionalEncoding(embedding_size, dropout=dropout)
        encoder_layer = TransformerEncoderLayer(embedding_size, num_heads, dropout=dropout)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers)
        self.pool = nn.AvgPool2d(2)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hpe = self.embed(x)
        hpe = self.pos_embed(hpe)
        h = self.encoder(hpe)
        h = self.pool(h)
        return h

# Initializing the model
embedding_size = 64
num_heads = 4
num_encoder_layers = 3
dropout = 0.1
m = Transformer(embedding_size, num_heads, num_encoder_layers, dropout)

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
