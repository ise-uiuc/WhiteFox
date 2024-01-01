
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, 3, stride=2, padding=2)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 - 2.0
        v3 = F.relu(torch.squeeze(v2, 0))
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
