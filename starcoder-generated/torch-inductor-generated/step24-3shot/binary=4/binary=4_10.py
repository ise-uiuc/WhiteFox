
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(3, 8)
        self.linear2 = torch.nn.Linear(3, 8)
 
    def forward(self, x1):
        v1 = self.linear1(x1)
        v2 = v1 + 2
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
