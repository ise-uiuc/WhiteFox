
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(4, 12, 3, stride=4, padding=2)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 - 1
        v3 = F.relu(v2)
        v4 = torch.add(v3, x1)
        v5 = v4.view(1, 48, 10, 10)
        return v5
# Inputs to the model
x1 = torch.randn(1, 4, 26, 26)
