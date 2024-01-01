
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 9)
        self.linear2 = torch.nn.Linear(9, 10)
 
    def forward(self, x1, x2):
        v1 = self.linear1(x2)
        v2 = v1 + x1
        v3 = self.linear2(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 10)
x2 = torch.randn(1, 10)
