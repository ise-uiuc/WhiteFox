
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, x3):
        v1 = torch.cat([x1, x2, x3], dim=1)
        v2 = v1[:, 0:8388607]
        v3 = v2[:, 0:700]
        v4 = torch.cat([v1, v3], dim=1)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 700, 128, 128)
x2 = torch.randn(1, 8898471, 102, 102)
x3 = torch.randn(1, 9223372036854775807 - 12800, 128, 128)
