
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 4)
 
    def forward(self, x1):
        x1 = self.linear(x1)
        x2 = x1 > 0
        x3 = x1 * 0.25
        x4 = torch.where(x2, x1, x3)
        return x4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 2)
