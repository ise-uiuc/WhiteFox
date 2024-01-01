
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1):
        v1 = torch.cat([x1, x1])
        v2 = v1[:, None, :, :, None]
        v3 = v2[0, :, :, None]
        v4 = [v1, v3]
        v5 = torch.cat(v4, 2)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 5, 3, 224, 224)
