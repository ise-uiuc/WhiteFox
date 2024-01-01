
class Model(torch.nn.Module):
    def __init__(self, m): # m is an additional tensor
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
 
    def forward(self, x1, m):
        v1 = self.conv(x1)
        v2 = v1 + m
        return v2

# Initializing the model with placeholder tensor
m = Model(torch.randn(1, 8, 64, 64))

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
