
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 1000)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        yhat = torch.tanh(v1)
        return yhat

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8)
