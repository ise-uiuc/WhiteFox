
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool = torch.nn.MaxPool2d(3, 2, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.maxpool(x1)
        v2 = v1 - 100
        v3 = F.relu(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
