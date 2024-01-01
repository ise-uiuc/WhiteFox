
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
    def forward(self, x1, x2, x3):
        v1 = self.conv1(x1)
        v2 = torch.bmm(x2, v1.reshape(16, -1).unsqueeze(2))
        v3 = self.pool2d(v2.reshape(-1, 32, 128))
        return v3
# Inputs to the model
x1 = torch.randn(16, 16, 64, 64)
x2 = torch.randn(16, 16, 64, 128).permute(2, 0, 1).unsqueeze(0)
x3 = torch.randn(16, 16, 64, 128).permute(2, 0, 1).unsqueeze(0)
