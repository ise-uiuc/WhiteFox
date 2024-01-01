
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(8, 8)
        self.linear2 = torch.nn.Linear(8, 8)
 
    def forward(self, x):
        v1 = self.linear1(x)
        v2 = self.linear2(v1)
        return v1 + v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(8)
