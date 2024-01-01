
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(8, 4, 5, stride=2, padding=2, dilation=1)
        self.relu = torch.nn.ReLU()
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.relu(v1)
        v2 = v2.mean(dim=1, keepdim=True)
        return v2
# Inputs to the model
x1 = torch.randn(1, 8, 64, 64)
