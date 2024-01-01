
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, Q, K, V, mask):
        s = torch.matmul(Q, K.transpose(-2, -1))
        b = s * mask
        c = torch.softmax(b, dim=-1) * mask
        a = torch.matmul(c, V)
        return a
# Inputs to the model
q1 = torch.randn(1, 64, 56, 56)
k3 = torch.randn(1, 64, 56, 56)
v3 = torch.randn(1, 64, 56, 56)
mask = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
