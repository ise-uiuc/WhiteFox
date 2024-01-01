
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = torch.nn.MaxPool2d(2, padding=1, stride=1)
    def forward(self, x1):
        v1 = self.pool(x1)
        v2 = v1 + v1
        return v2
# Inputs to the model
x1 = torch.randn(1, 5, 64, 64)
