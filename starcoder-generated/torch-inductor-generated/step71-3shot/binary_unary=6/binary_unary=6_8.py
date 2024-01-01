
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8, bias=True)
 
    def forward(self, x):
        y = self.linear(x)
        z = y - 0.4
        return z

# Initializing the model
m = Model()

# Input to the model
x = torch.randn(1, 3)
