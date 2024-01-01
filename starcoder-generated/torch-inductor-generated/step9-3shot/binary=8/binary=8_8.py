 
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, hidden_size, kernel_size=1, groups=8)
        
    def forward(self, x, other=None):
        v1 = self.conv(x)
        if other is not None:
            v2 = v1 + other
        else:
            v2 = v1
        
        return v2

# Initializing the model
m = Model()
  
# Inputs to the model
x = torch.randn(1, 3, 64, 64)

# This code might result in an error
other = torch.randn(8, 3, 1, 1)
