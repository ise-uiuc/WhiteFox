
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 16, 3, stride=1, padding=1)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = torch.relu(v1)
        ret = torch.sum(v2, dim=-1)
        return ret
# Inputs to the model
x = torch.randn(1, 16, 64, 64)
