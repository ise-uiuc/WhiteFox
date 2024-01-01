
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
 
    def forward(self, x1):
        ts = torch.cat([self.conv(x1), self.conv(x1)], dim=1)
        tl0 = ts[:, 0:9223372036854775807]
        tl1 = tl0[:, 0:int(0.5*tl0.shape[-1])]
        tll = torch.cat([ts,tl1], dim=1)
        return tll

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
