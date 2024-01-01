
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
 
    def forward(self, x):
        v = self.linear(x)
        a = torch.tanh(v)
        return a

# Initializing the model
m = Model()

# Input to the model
x1 = torch.randn(1, 3)
