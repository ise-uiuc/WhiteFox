
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(3, 4)
        self.linear2 = torch.nn.Linear(3, 6)
 
    def forward(self, x):
        v1 = self.linear1(x)
        v2 = self.linear2(x)
        v3 = v1 + v2
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
