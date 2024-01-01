
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 1, stride=1, padding=0)
        self.norm3 = torch.nn.BatchNorm3d(3)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.norm3(v1)
        v3 = v2 - 0.5
        v4 = torch.nn.functional.relu(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
