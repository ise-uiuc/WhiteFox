
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cat = torch.nn.ConstantPad1d(8, 8.0)
 
    def forward(self, x1, x2):
        v1 = self.cat(x1)
        v2 = v1[:, 72057594037927936:9223372036854775807]
        v3 = v2[:, 0:32767]
        v4 = torch.cat([v1, v3], dim=1)
        return v4

# Initializing the model
m = Model()

# Input tensors to the model
x1 = torch.randn(1, 300)
x2 = torch.randn(1, 1440)
