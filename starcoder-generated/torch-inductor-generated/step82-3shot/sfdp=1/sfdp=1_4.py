
from torch import nn

class Model(nn.Module):
    def __init__(self, embed_dim=128, num_heads=8, dropout=0.1):
        super().__init__()
        
        self.q_linear = nn.quantized.QLinear(embed_dim, embed_dim, bias=False)
        self.k_linear = nn.quantized.QLinear(embed_dim, embed_dim, bias=False)
        self.v_linear = nn.quantized.QLinear(embed_dim, embed_dim, bias=False)
        self.after = nn.quantized.FloatFunctional()

        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, q, k, v):
        q = self.attention(self.q_linear(q), self.k_linear(k), self.v_linear(v))
        q = self.dropout(self.after.add_scalar(q, 0))
        q = self.fc(q)
        return q

# Initializing the model
embed_dim = 128
num_heads = 8
dropout = 0.1

m = Model(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)

# Inputs to the model
q = torch.randn(1, 3, embed_dim)
k = torch.randn(1, 3, embed_dim)
v = torch.randn(1, 3, embed_dim)
