
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=2, padding=1)
 
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1.permute([0, 2, 3, 1])
        v3 = torch.nn.functional.pad(v2, [0, 0, 1, 0], "constant", 1.0)
        v4 = v3.permute([0, 3, 1, 2])
        v5 = torch.nn.functional.pad(v4, [0, 0, 0, 0, 1, 0], "constant", 0.0)
        v6 = torch.nn.functional.pad(v5, [0, 1, 0, 0], "constant", 0.0)
        v7 = torch.nn.functional.pad(v6, [0, 1, 0, 0, 0, 0], "constant", 1.0)
        v8 = v7[:, 1:, :, :]
        v9 = torch.cat([v1, v8], 1)
        return v9

# Initializing the model
m = Model()

# Input to the model
x = torch.randn(1, 3, 256, 256)
