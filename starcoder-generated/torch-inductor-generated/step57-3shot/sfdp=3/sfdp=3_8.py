
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v):
        qk = torch.matmul(q, k.transpose(-2, -1))
        s = torch.tensor(1 / math.sqrt(k.size(-1)))
        softmax_qk = qk.mul(s).softmax(dim=-1)
        return softmax_qk.matmul(v)

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(2, 10, 4)
k = torch.randn(2, 8, 4)
v = torch.randn(2, 8, 5)
