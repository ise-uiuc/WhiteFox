
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1536, 256, 1, stride=1)
 
    def forward(self, x):
        v1 = self.conv(x)
        v2 = torch.cat([v1, v1**0.5], 1)
        v3 = torch.cat([v2, v2[1:]])
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 1536, 64, 64)
