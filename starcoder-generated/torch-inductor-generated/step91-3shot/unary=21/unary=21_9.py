
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 14, 11, stride=9, padding=5)
        self.conv2 = torch.nn.Conv2d(14, 3, 9, stride=8, padding=3)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = torch.tanh(v1)
        v3 = self.conv2(v2)
        return v3
# Inputs to the model
x = torch.randn(1, 3, 297, 348)
