
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 15, 3, stride=5)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 - 0.48
        v3 = F.relu(v2)
        v4 = torch.squeeze(v3, 2)
        return v4
# Inputs to the model
x1 = torch.randn(1, 1, 25, 25)
