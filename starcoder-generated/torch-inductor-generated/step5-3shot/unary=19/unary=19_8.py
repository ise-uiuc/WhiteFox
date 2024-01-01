
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(32, 1)
 
    def forward(self, x):
        y = self.linear(x)
        # y is assumed to be a float in [-30, 30]
        yexp = y.exp()
        return yexp / (1 + yexp)

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(16, 32)
