
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        input_channels = 1024
        self.linear = torch.nn.Linear(input_channels, 2048)
 
    def forward(self, x1):
        z1 = self.linear(x1)
        z2 = z1 + z1
        return z2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 1024)
