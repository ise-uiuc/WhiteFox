
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 128, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1, 3).contiguous()
        v2 = self.conv1(v1)
        v3 = v2.permute(0, 2, 1, 3).contiguous()
        return v3
# Inputs to the model
x1 = torch.randn(1, 1, 32, 32)
