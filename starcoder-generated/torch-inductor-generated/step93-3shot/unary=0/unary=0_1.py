
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(32, 54, 7, stride=1, padding=2)
    def forward(self, x11):
        v1 = self.conv(x11)
        v2 = 0.9999996423721313 * v1
        v3 = torch.flatten(v2, 1)
        v4 =   v3 +   0.33320503807067873
        v5 = v4 * torch.sin(v3)
        v6 = torch.cos(v2)
        v7 = v5 * v6 *  0.66666626929626465
        return v7
# Inputs to the model
x11 = torch.randn(1, 32, 19, 55)
