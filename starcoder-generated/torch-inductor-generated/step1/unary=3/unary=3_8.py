
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(12, 24, 3, stride=2, padding=0, padding_mode="circular")
 
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 * 9.536806187858077e-7
        v3 = v1 * 0.2569044548940507
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        return v6

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 12, 24, 24)
