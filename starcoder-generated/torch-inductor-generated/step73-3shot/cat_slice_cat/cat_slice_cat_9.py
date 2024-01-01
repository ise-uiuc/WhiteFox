 2
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
 
    def forward(self, x1, x2):
        # concat
        v0 = torch.cat([x1, x2], 1)
        v1 = self.conv(v0)
        # split
        size = v1.shape[1] // 2
        v2 = v1[:, :size]
 
        # concat
        v3 = torch.cat([v1, v2], 1)
        v4 = v3[:, 0:size]
 
        # concat
        v5 = torch.cat([v0, v4], 1)
        v6 = v5[:, 0]
        return v6
 
# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 64, 64)
