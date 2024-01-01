
class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10,10)
 
    def forward(self, x1, x2):
        return self.linear(x1) + x2

# Initializing the model
m = M()

# Inputs to the model
x1 = torch.randn(1, 10)
x2 = torch.randn(1, 10)
