
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lm = torch.nn.Linear(3, 4)
 
    def forward(self, x1, other):
        return self.lm(x1) + other

# Initializing the model
m = Model()

# Inputs to the model; x1 is also given another tensor as a keyword argument.
x1 = torch.randn(1, 3)
other = torch.randn(1, 3)
