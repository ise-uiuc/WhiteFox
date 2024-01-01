
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU(inplace=True)
    def forward(self, x, inp):
        v1 = self.relu(x)
        v2 = v1.mean()
        v3 = torch.mm(x, x)
        v4 = torch.matmul(v3, v3) + v2
        v5 = v4 + inp
        return v5
# Inputs to the model
x = torch.randn(3, 3)
inp = torch.randn(3, 3)
