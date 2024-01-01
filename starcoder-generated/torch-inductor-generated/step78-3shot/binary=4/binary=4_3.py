
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = torch.nn.Linear(48, 5)
 
    def forward(self, x1, another):
        v1 = self.lin1(x1)
        v2 = v1 + another
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 48)
another = torch.randn(2, 5)
