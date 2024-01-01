
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv3d(1, 20, 3)
        self.conv2 = torch.nn.Conv3d(20, 20, 3, padding=(1, 1, 1))
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = v2 - 1
        v4 = F.relu(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64, 64)
