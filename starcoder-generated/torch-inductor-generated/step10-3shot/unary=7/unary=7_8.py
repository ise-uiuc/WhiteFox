
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.actv = torch.nn.SELU()
 
    def forward(self, x1):
        b1 = x1
        b2 = self.actv(x1)
        b3 = b1 + b2
        b4 = b3 / 6.0
        return b4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
# Setting min and max to 0
