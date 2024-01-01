
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv3d(1, 10, (2, 3, 4), stride=2, padding=2)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 - '42.0'
        return v2
# Inputs to the model
x = torch.randn(1, 1, 3, 4, 5)
