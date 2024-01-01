
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x):
        x1 = x[:, 0:size]
        x2 = torch.cat([x, x1], dim=1)
        return x2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64, 64)
x2 = torch.randn(1, 3, 32, 32, 32)
