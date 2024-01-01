
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        heads = 8
        d_feat = 64
        self.query = torch.nn.Linear(d_feat, heads * d_feat)
        self.key = torch.nn.Linear(d_feat, heads * d_feat)
        self.value = torch.nn.Linear(d_feat, heads * d_feat)
        self.layernorm1 = torch.nn.LayerNorm(heads * d_feat, eps=1e-5)
        self.linear1 = torch.nn.Linear(heads * d_feat, 4 * heads * d_feat)
        self.linear2 = torch.nn.Linear(4 * heads * d_feat, heads * d_feat)
        self.layernorm2 = torch.nn.LayerNorm(d_feat, eps=1e-5)
 
    def forward(self, x1, x2, attn_mask):
        v1 = self.layernorm1(x1)
        v2 = self.query(v1).view(v1.shape[:-1] + (4, -1)).transpose(-2, -3)
        v3 = self.key(x2).view(v1.shape[:-1] + (4, -1))
        v4 = self.value(x2)
        v5 = v3 @ torch.transpose(v2, -2, -3) / math.sqrt(v3.size(-1))
        v6 = v5 + attn_mask
        v8 = torch.softmax(v6, dim=-3)
        v9 = torch.matmul(v9, v4)
        v10 = self.linear2(self.linear1(torch.flatten(v9, start_dim=-2)))
        v12 = self.layernorm2(v10 + v1)
        return v12

# Initializing the model
m = Model()

# Inputs to the model
n_samples = 1
d_feat = 64
x1 = torch.randn(n_samples, d_feat, d_feat)
x2 = torch.randn(n_samples, 4, d_feat, d_feat)
attn_mask = torch.randint(831, (n_samples, 8, 4, 4,), dtype=torch.float32)
