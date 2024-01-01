
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(5, 5, bias=False)
        self.other = torch.rand(5, 5)
 
    def forward(self, x1):
        return self.linear(x1) + self.other

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 1, 5)
