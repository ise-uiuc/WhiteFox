
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 5, stride=3, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 - 1.0
        v3 = torch.nn.functional.relu(v2)
        v4 = torch.squeeze(v3, 0)
        return v4
# Inputs to the model
x1 = [torch.randn(1, 3, 64, 64), torch.randn(1, 3, 64, 64)]
