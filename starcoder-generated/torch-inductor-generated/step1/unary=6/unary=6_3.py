
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 8, 3, stride=1, padding=1)
 
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = v1 + 3
        v3 = v2.clamp_min(0).clamp_max(6)
        v4 = self.conv2(v3)
        v5 = v4 * 6
        return v5


# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3, 64, 64)
