
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = torch.nn.MaxPool2d(2, stride=2)
        self.relu1 = torch.nn.ReLU()
        self.conv = torch.nn.Conv2d(512, 768, 2, stride=2)
    def forward(self, x):
        v1 = self.pool(x)
        v2 = self.relu1(v1)
        v3 = self.conv(v2)
        v4 = torch.tanh(v3)
        return v4
# Inputs to the model
x = torch.randn(1, 512, 28, 28)
