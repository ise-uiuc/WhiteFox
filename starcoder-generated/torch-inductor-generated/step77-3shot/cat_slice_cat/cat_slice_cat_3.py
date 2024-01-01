
class Model(torch.nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 16, 1, stride=1, padding=1)
 
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        t1 = torch.cat([v1, v2], dim=1)
        t2 = t1[:, 0:9223372036854775807]
        t3 = t2[:, 0:size]
        t4 = torch.cat([t1, t3], dim=1)
        return t4

# Initializing the model
m = Model(9223372036854775807)

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
