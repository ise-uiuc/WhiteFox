
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = torch.nn.Conv2d(24, 32, 1, stride=1, padding=0)
        self.conv1 = torch.nn.Conv2d(32, 16, 3, stride=1, padding=1)
    def forward(self, x1, x2, x3):
        v1 = self.conv0(torch.cat((x1, x2, x3), dim=1))
        ret0 = v1 - 1.0
        ret0 = F.relu(ret0)
        v1 = self.conv1(ret0)
        v2 = v1 + 0.5
        v3 = F.relu(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 24, 64, 64)  # This shape triggers ReLU_1->ReLU6.
x2 = torch.randn(1, 24, 64, 64)  # This shape triggers ReLU_1->DepthwiseConv2D->BatchNorm2d.
x3 = torch.randn(1, 24, 64, 64)  # This shape triggers ReLU_1->Conv2D->BatchNorm2d.
