
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 8)
 
    def forward(self, x1):
        b1 = self.linear(x1)
        return b1 + x1

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 2)
