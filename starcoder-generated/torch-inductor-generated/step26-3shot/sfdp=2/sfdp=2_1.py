
class Model(torch.nn.Module):
    def __init__(self, num_attention_heads, dim_qkv, dropout=[]):
        super().__init__()
        self.query = torch.nn.Linear(dim_qkv, dim_qkv, bias=True)
        self.key = torch.nn.Linear(dim_qkv, dim_qkv, bias=False)
        self.value = torch.nn.Linear(dim_qkv, dim_qkv, bias=False)
        self.softmax_qk = torch.nn.Softmax(dim=-1)
        self.dropout = torch.nn.Dropout(dropout)
 
    def forward(self, x1, x2):
        v1 = self.query(x1)
        v2 = self.key(x2)
        v3 = torch.matmul(v1, v2.transpose(-2, -1))
        __inv_scale_factor__ = 1.0 / math.sqrt(dim_qkv / num_attention_heads)
        v4 = v3.div(__inv_scale_factor__)
        v5 = self.softmax_qk(v4)
        v6 = self.dropout(v5)
        v7 = torch.matmul(v6, self.value(x2))
        return v7

# Initializing the model
m = Model(num_attention_heads=8, dim_qkv=16, dropout=0.0)

# Inputs to the model
x1 = torch.randn(1, 1, 16)
x2 = torch.randn(1, 8, 16)
