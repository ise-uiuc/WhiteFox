
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 12, 1, stride=1)
        self.pool = torch.nn.MaxPool2d(7, stride=3, padding=0)
    def forward(self, x):
        v1 = self.conv1(x)
        v1 = torch.tanh(v1)
        v1 = self.pool(v1)
        return v1
# Inputs to the model
x = torch.randn(64, 3, 224, 224)
