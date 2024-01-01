
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = torch.nn.Conv2d(3, 69, 1, stride=2)
        self.conv1 = torch.nn.Conv2d(69, 63, 1, stride=2)
    def forward(self, x1):
        v1 = self.conv0(x1)
        v10 = self.conv1(v1)
        v11 = v10 - 230
        v12 = F.relu(v11)
        return v1
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
