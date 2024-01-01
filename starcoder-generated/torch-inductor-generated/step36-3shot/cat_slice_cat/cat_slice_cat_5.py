
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2):
        v1 = torch.cat([x1, x2], dim=1)
        v2 = v1[:, 1:1073741823]
        v3 = v2[:, 1:int64.MaxValue]
        v4 = torch.cat([v1, v3], dim=1)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 922337203685477580, 256, 256)
x2 = torch.randn(1, 1073741820, 256, 256)
