
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convA = torch.nn.Conv2d(3, 8, 1)
        self.convB = torch.nn.Conv2d(3, 16, 1)
        self.convC = torch.nn.Conv2d(3, 32, 1)
        self.convD = torch.nn.Conv2d(3, 64, 1)
    def forward(self, x1):
        v1 = self.convA(x1)
        v2 = self.convB(x1)
        v3 = self.convC(x1)
        v4 = self.convD(x1)
        x = torch.tanh(v1 + v2 + v3 + v4)
        return x
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
