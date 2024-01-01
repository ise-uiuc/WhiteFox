
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2):
        s1, s2, s3 = torch.split(x1, [1,1,1], 0)
        c = torch.cat([s1, s2, s3], dim=0)
        return c

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(3, 1, 64, 64)
x2 = torch.randn(3, 1, 64, 64)
