
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(100,50)
 
    def forward(self, x1):
        v1 = self.conv(x1)
        return v1

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(100)
