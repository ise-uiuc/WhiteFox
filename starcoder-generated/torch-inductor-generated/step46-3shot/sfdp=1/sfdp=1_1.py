
class Model(torch.nn.Module):
    def __init__(self, embed_dim, num_heads, dim_feedforward, dropout_p):
        super().__init__()
        self.qkv = torch.nn.Linear(embed_dim, 3 * embed_dim)
        self.linear1 = torch.nn.Linear(3 * embed_dim, dim_feedforward)
        self.dropout = torch.nn.Dropout(p=dropout_p)
        self.linear2 = torch.nn.Linear(dim_feedforward, embed_dim)
 
    def forward(self, x1):
        qkv = self.qkv(x1)
        q, k, v = qkv.chunk(3, dim=-1)
        output = self.dropout((q * k) / math.sqrt(k.size(-1)))
        output = self.dropout(self.linear2(self.dropout(self.linear1(output))))
        return output

# Initializing the model
m = Model(embed_dim=16, num_heads=4, dim_feedforward=16, dropout_p=0.5)

# Inputs to the model
x1 = torch.randn(1, 1, 4096)
