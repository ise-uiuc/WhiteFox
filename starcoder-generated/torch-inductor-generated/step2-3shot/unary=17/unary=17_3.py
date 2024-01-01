
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv1 = torch.nn.ConvTranspose2d(8, 8, 3, stride=1, padding=0)
    def forward(self, x1_1):
        v1 = self.conv(x1_1)
        v2 = v1.view(1, 8, 2, 2)
        v3 = self.conv1(v2)
        v4 = F.relu(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 8, 8)
