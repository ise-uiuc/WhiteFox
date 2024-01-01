
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(6, 8, 2, stride=2)
        self.conv2 = torch.nn.Conv2d(8, 16, 2, stride=2)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = F.relu(v1)
        return self.conv2(v2)
# Inputs to the model
x = torch.randn(128, 6, 77, 87)
