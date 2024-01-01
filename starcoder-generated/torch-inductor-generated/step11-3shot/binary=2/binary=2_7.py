
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x3):
        v3 = self.conv(x3)
        v4 = v3 - torch.ones((1, 1, 55, 44), dtype=torch.float, device=x3.device)
        return v4
# Inputs to the model
x3 = torch.randn(1, 3, 88, 77)
