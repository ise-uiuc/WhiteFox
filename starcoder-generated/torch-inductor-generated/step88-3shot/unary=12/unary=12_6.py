
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 6, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = F.relu(x1)
        v2 = self.conv(v1)
        v3 = F.sigmoid(v2)
        v4 = F.tanh(x1.permute(0, 1, 3, 2))
        v5 = torch.mean(self.conv(v4), 1) + v2
        return v2.permute(0, 2, 3, 1) + v5.permute(0, 2, 1, 3) + v5
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
