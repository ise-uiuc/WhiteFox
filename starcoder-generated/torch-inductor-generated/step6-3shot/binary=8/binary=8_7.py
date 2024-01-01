
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
 
    def forward(self, x1, other=None):
        if not other:
            other = torch.randn(8, 3, 1, 1) # The shape of the other tensor to be added
        v1 = self.conv(x1)
        v2 = v1 + other
        return v2

# Initializing the model
m = Model()

# Initializing the second input tensor
other = torch.randn(8, 3, 1, 1)

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
