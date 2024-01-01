 definition
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 1)
 
    def forward(self, x):
        y = x.view(x.size(0), -1)
        y = self.linear(y)
        y = y * torch.clamp(y + 3, min=0, max=6) / 6
        return y 

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 16)
