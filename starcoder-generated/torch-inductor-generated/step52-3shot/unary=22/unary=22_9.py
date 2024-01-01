
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(5, 10)
 
    def forward(self, x):
        h = self.linear(x)
        return F.tanh(h)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 5)
