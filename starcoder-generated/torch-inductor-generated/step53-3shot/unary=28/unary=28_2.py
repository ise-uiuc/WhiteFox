
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(5, 3)
 
    def forward(self, x):
        y = self.linear(x).clamp(min=-0.5, max=0.5)
        return y

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(2, 5)
