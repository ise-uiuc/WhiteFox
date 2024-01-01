
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 2, 10, stride=5, padding=5)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 - 0.5
        v3 = F.relu(v2)
        v4 = torch.transpose(v3, 0, 1)
        return v4
# Inputs to the model
x1 = torch.randn(1, 1, 16, 16)
