
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 3, True)
 
    def forward(self, x0):
        v1 = self.linear(x0)
        v2 = v1 - torch.tensor([0, 2, 1])
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x0 = torch.randn(4, 10)
