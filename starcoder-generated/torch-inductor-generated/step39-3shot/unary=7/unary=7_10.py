
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.max_i = torch.nn.InstanceNorm2d(3, affine=True)
 
    def forward(self, x):
        v1 = self.max_i(x)
        return v1

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3, 64, 64)
