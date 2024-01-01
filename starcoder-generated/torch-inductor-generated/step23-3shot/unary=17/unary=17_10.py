
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pointwise = torch.nn.ConvTranspose2d(3, 1, 1)
        self.linear = torch.nn.Linear(4304, 10)
    def forward(self, x1):
        v1 = self.pointwise(x1)
        v2 = v1.reshape([v1.size()[0], -1])
        v3 = self.linear(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 28, 28)
