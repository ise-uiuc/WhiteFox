
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 12, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = torch.nn.Conv2d(12, 12, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = torch.nn.Conv2d(12, 3, kernel_size=3, stride=1, padding=1, bias=False)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = torch.tanh(v1)
        v3 = self.conv2(v2)
        v4 = torch.tanh(v3)
        v5 = self.conv3(v4)
        v6 = torch.tanh(v5)
        return v5
# Inputs to the model
x = torch.randn(5, 3, 224, 224)
