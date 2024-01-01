
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(256, 100)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, 0.00392156862745098) # Clamp the output of the linear transformation to a minimum value 2.0/255
        v3 = torch.clamp_max(v2, 0.996078431372549) # Clamp the output of the previous operation to a maximum value 1-(2.0/255)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 256)
