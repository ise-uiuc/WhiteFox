
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        
    def forward(self, x):
        x = self.conv(x)
        return x

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3, 64, 64)
