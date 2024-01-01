
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(2, 3, 1, stride=1, padding=1)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = v1 * 0.1
        v3 = v1[0, 0, 0, :]
        v4 = v1[:, :, 0, 0]
        v5 = torch.cat((v3, v4), dim=0)
        return v5
# Inputs to the model
x1 = torch.randn(1, 2, 64, 64)
