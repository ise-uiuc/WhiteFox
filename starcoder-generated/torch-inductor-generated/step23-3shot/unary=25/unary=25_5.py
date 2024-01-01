
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 5)
    
    def forward(self, x1):
        x2 = self.linear(x1)
        x3 = x2 > 0
        x4 = x2 * 0.02
        x5 = torch.where(x3, x2, x4)
        return x5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
