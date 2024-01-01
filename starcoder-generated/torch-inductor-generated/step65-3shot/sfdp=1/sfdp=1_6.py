
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x):
        q = x.reshape(x.shape[0], 1, -1)
        k = x.reshape(x.shape[0], -1, 1)
        v = q * k
        return v

# Initializing the model
m = Model()

# Input to the model
x = torch.randn(2, 3*4)
