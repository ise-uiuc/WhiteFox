 
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(100, 100)

    def forward(self, x1, x2):
        result = torch.where(x1 > 0, x1+x2/2, 0.5*x2)
        return result

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 100)
x2 = torch.randn(1, 100)
