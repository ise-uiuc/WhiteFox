
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 64, 3, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 - 10
        v3 = F.relu(v2)
        v4 = v3.unsqueeze(0)
        return v4
# Inputs to the model
x1 = torch.randn(1, 16, 56, 56)
