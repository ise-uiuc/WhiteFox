
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.dense2 = torch.nn.Linear(100, 100)
    def forward(self, x, x2, z, m4):
        qk = x @ x2.transpose(-2, -1) / math.sqrt(x.size(-1))
        qk = qk + m4
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ z
        return output
# Inputs to the model
x = torch.randn(1, 1024, 14, 14)
x2 = torch.randn(1, 1024, 14, 14)
z = torch.randn(1, 1024, 14, 14)
mask = (torch.rand(1, 14, 14) > 0.7).fill_(-1000000000.0)
