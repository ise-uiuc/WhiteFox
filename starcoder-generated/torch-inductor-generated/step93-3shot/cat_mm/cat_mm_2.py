
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v = torch.matmul(x1, x2)
        t = torch.cat([v] + 100*[v])
        return torch.cat([t] + 100*[t], 1)
# Inputs to the model
x1 = torch.randn(1, 1)
x2 = torch.randn(1, 1)
