
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 59, 2, stride=6, padding=0)
        self.relu = torch.nn.Tanh()
        a = torch.tensor([-1., -2, -3])
        self.a = torch.nn.Parameter(a, requires_grad=True)
        b = torch.tensor([6., -7, 8.])
        self.b = torch.nn.Parameter(b, requires_grad=False)
    def forward(self, x81):
        v1 = self.conv(x81)
        v2 = self.a * (v1 + x81) + torch.matmul(x81, F.relu(self.b))
        return v2
# Inputs to the model
x81 = torch.randn(1, 4, 29, 9)
