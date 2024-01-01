
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 1)
    def forward(self, x1):
        x2 = self.conv(x1)
        x3 = x2 - 0.2
        x4 = x3.permute(0, 1, 3, 2).contiguous()
        x5 = F.relu(x4)
        return x5
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
