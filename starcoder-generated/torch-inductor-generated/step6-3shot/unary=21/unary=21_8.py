
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=2, padding=1, dilation=2)
        self.relu = torch.nn.ReLU(inplace=False)
    def forward(self, x):
        v1 = self.conv(input)
        v2 = self.relu(v1)
        v3 = torch.tanh(v2)
        return v3
# Inputs to the model
input = torch.randn(1, 3, 256, 256)
