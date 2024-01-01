
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 4, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(4, 4, 1, stride=1, padding=1)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = self.conv2(v1)
        v3 = v2 - v1
        return v3
# Inputs to the model
x = torch.randn(1, 1, 1, 1, dtype=torch.float32, device = 'cpu', requires_grad = True)
