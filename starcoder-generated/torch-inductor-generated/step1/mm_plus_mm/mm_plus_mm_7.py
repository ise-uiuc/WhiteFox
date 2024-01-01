
class Model(torch.nn.Module):
    def __init__(self, arg):
        super().__init__()
        self.arg = arg
 
    def forward(self, x):
        v1 = self.arg
        v2 = torch.mm(x, self.arg)
        v3 = torch.mm(self.arg, v2)
        return v3 + v2

# Initializing the model
m = Model(torch.randn(128, 64))

# Inputs to the model
x = torch.randn(1, 64, 1)
