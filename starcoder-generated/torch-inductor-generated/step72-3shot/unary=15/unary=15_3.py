
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(32, 32, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1.transpose(2, 3).transpose(1, 2).contiguous()
        return v2
# Inputs to the model
x1 = torch.randn(1, 64, 96, 64)
