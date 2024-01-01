
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(4, 32, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 16, 1, stride=1, padding=1)
    def forward(self, x1, x2):
        out_x1 = self.conv1(x1)
        out_x2 = self.conv1(x2)
        out = torch.cat([out_x1, out_x2], dim=1)
        out = self.conv2(out)
        return out
# Inputs to the model
x1 = torch.randn(1, 4, 32, 32)
x2 = torch.randn(1, 8, 32, 32)
