
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(6, 8, 1, stride=1, padding=1)
 
    def forward(self, x1, x2):
        v1 = self.conv1(x1)
        v11 = self.conv2(x2)
        v2 = torch.split(v1 + v11, 1, dim=1)
        v3 = torch.cat([v2[7], v2[0], v2[1], v2[2], v2[6], v2[5], v2[3], v2[4]], dim=1)
        v4 = torch.cat([v2[6], v2[7], v2[4], v2[5], v2[0], v2[1], v2[2], v2[3]], dim=1) 
        v5 = torch.cat([v2[5], v2[6], v2[3], v2[4], v2[7], v2[0], v2[1], v2[2]], dim=1)
        v6 = torch.cat([v2[1], v2[2], v2[5], v2[6], v2[7], v2[4], v2[3], v2[0]], dim=1) 
        v7 = torch.cat([v2[2], v2[1], v2[6], v2[5], v2[4], v2[3], v2[0], v2[7]], dim=1)
        v8 = torch.cat([v2[3], v2[2], v2[7], v2[6], v2[5], v2[4], v2[1], v2[0]], dim=1)
        v9 = torch.cat([v2[4], v2[3], v2[0], v2[7], v2[6], v2[1], v2[2], v2[5]], dim=1) 
        return v3 + v4 + v5 + v6 + v7 + v8 + v9 + v1 + v11

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 6, 64, 64)
