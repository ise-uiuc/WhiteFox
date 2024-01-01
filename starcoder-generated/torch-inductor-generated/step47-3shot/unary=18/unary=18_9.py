
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=4, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.sigmoid(v1)
        v3, v4 = torch.max(v2, dim=1, keepdim=False)
        v5 = torch.sigmoid(v3)
        v6 = torch.pow((v3), 2.0)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
