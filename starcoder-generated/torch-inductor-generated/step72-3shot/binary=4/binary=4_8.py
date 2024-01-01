
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 5)
        self.linear2 = torch.nn.Linear(5, 2)
 
    def forward(self, x):
        v1 = self.linear1(x)
        v2 = v1 + x
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(16, 2)
