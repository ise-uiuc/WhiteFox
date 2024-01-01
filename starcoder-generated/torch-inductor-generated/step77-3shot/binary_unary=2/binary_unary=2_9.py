
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(7, 3, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.nn.functional.relu(v1)
        v3 = v2 - 0.1
        return v3
# Inputs to the model
x1 = torch.randn(1, 7, 20, 20)
