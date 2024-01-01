
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(8, 8)
        self.linear2 = torch.nn.Linear(8, 8)
 
    def forward(self, x1, x2):
        v1 = self.linear1(x1)
        v2 = v1.add(x2)
        v3 = self.linear2(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 1, 1024)
x2 = torch.randn(1, 1024)
