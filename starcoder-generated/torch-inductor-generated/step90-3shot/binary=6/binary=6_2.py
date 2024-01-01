
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(32, 64)
        self.linear2 = torch.nn.Linear(32, 1)
 
    def forward(self, x1):
        v1 = self.linear1(x1)
        v2 = v1 - x1
        v3 = self.linear2(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
_input = torch.randn(1, 32)
other = torch.randn(1, 1)
