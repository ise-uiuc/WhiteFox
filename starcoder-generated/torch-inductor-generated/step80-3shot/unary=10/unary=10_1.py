
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)
 
    def forward(self, x0):
        v0 = torch.nn.functional.relu6(x0 + 3)
        return v0 / 6

# Initializing the model
m = Model()

# Inputs to the model
x0 = torch.randn(1, 4)
