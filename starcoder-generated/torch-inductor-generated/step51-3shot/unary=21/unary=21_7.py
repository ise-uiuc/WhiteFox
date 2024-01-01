
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 8, 1)
        self.conv3 = torch.nn.Conv2d(8, 32, 1)
        self.tanh = torch.nn.Tanh()
    def forward(self, x):
        v1 = self.conv1(x)
        v1 = self.tanh(v1)
        v1 = self.conv2(v1)
        v1 = self.tanh(v1)
        v1 = self.conv3(v1)
        v1 = self.tanh(v1)
        return v1
# Inputs to the model
x = torch.randn(1, 3, 224, 224)
