
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lineartanh = torch.nn.Sequential(
            torch.nn.Linear(780, 64),
            torch.nn.Tanh())
 
    def forward(self, x):
        v = self.lineartanh(x)
        return v

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 780)
