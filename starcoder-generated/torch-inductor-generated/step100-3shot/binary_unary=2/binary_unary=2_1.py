
class Model(torch.nn.Module):
    def __init__(self):
#         super().__init__()
        self.conv = torch.nn.Conv2d(64, 128, (7, 6), stride=2, padding=(0, 2))
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = F.relu(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 64, 128, 64)
