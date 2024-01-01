
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(5, 1)
 
    def forward(self, x):
        t1 = self.linear(x)
        t2 = torch.tanh(t1)
        return t2

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 5)
