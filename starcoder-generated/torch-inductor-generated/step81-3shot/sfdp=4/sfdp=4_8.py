
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dense1 = torch.nn.Linear(100, 100)
        self.activation = torch.nn.ReLU()
    def forward(self, x, y, z, mask):
        qk = x @ y.transpose(-2, -1) / math.sqrt(x.size(-1))
        qk = qk + mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ z
        return output
# Inputs to the model
x = torch.randn(1, 2048, 14, 14)
y = torch.randn(1, 2048, 14, 14)
z = torch.randn(1, 2048, 14, 14)
mask = (torch.rand_like(x).ge(0.5)).to_sparse().to_dense().to(torch.float32) * -1e20
