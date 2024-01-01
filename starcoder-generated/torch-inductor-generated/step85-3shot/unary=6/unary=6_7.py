
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 16, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv(x1)
        d0 = v1.squeeze(0)
        e0 = v1.unsqueeze(0)
        return e0
# Inputs to the model
x1 = torch.randn(2, 3, 64, 64)
