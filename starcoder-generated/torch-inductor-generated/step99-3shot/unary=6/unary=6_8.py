
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1, stride=1, padding=0)
    def forward(self, x1):
        t1 = self.conv(x1)
        t2 = torch.clamp_min(2.5 * t1 + 2, 0)
        t3 = (t2 * t2)
        return (t3 * 0.8).unsqueeze(-1)
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
