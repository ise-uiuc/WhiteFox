
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = torch.nn.AvgPool2d(19)
    def forward(self, x1):
        v1 = self.pool(x1)
        v2 = torch.tanh(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 9, 100, 50)
