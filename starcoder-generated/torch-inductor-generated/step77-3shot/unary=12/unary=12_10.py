
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, (3, 3), stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.sigmoid(v1)
        v3 = (x1 * v2.type(x1.type()))
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 128, 128)
