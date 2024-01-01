
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 3, 3, stride=2)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = 0.001 * v1
        v3 = F.relu(v2)
        v4 = torch.mean(v3, 0, True)
        return v4
# Inputs to the model
x1 = torch.randn(1, 1, 128, 128)
