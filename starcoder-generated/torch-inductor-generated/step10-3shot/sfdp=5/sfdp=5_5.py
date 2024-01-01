
class Model(torch.nn.Module):
    def __init__(self, d_model, nhead, dropout_p=0.1):
        super().__init__()
        self.multihead_attn = torch.nn.MultiheadAttention(d_model, nhead, dropout_p)

    def forward(self, q, k, v, mask):
        return self.multihead_attn(q, k, v, mask)

# Initializing the model
nhead = 64
d_model = 64
dropout_p = 0.3
m = Model(d_model, nhead, dropout_p)

# Inputs to the model
q = torch.randn(2, 8, 64)
k = torch.randn(64, 8, 64)
v = torch.randn(64, 8, 64)
mask = None
