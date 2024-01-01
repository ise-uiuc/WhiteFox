
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(28*28, 10)
 
    def forward(self, x3):
        x2 = x3.view(x3.shape[0], 28*28)
        v4 = self.linear(x2)
        v2 = torch.tanh(v4)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x3 = torch.randn(1, 28, 28)
