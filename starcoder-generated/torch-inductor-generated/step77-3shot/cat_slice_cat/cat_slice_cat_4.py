
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2):
        v1 = torch.cat([x1, x2], dim=1)
        return v1[:, :, :, :34]

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 27, 3, 5)
x2 = torch.randn(1, 960, 3, 5)
