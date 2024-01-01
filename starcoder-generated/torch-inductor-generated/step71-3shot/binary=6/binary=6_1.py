
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 24)
 
    def forward(self, x1):
        l1 = self.linear(x1)
        l2 = l1 - 15.5
        return l2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(12, 16)
