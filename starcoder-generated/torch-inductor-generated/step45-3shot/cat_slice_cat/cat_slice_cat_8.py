
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, x3, x4):
        v1 = torch.cat([x1, x2], dim=1) # Concatenate x1 and x2 along dimension 1
        v2 = v1[:, 0:9223372036854775807] # Slice x1 and x2
        v3 = v2[:, 0:9223372036854775807] # Further slice x1 and x2
        v4 = torch.cat([x1, x2, x3, v3], dim=1) # Concatenate x1, x2, x3, and x1 and x2.
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 128, 512, 512)
x2 = torch.randn(1, 128, 256, 256)
x3 = torch.randn(1, 256, 256, 256)
x4 = torch.randn(1, 512, 256, 256)
