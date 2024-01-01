
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool = torch.nn.MaxPool2d(16, stride=8)
    def forward(self, x1):
        v1 = self.maxpool(x1)
        v2 = self.maxpool(x1)
        v3 = v1 + v2
        v4 = v3 + v2
        v5 = torch.relu(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 8, 64, 64)
