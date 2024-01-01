
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(9, 1)
 
    def forward(self, x1):
        x2 = torch.where((x1 < 0), torch.zeros(1), x1)
        x3 = x2 * 0.1
        x4 = self.linear(x3)
        return x4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 9)
