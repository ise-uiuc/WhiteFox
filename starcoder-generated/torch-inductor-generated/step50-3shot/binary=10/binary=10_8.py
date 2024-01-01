
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Linear(10, 50)
 
    def forward(self, x1, x2=tensor):
        v1 = self.conv(x1)
        v2 = v1 + x2
        return v2

# Initializing the model
m = Model()

tensor = torch.randn(50)

# Inputs to the model
x1 = torch.randn(10)
x2 = torch.randn(50)
