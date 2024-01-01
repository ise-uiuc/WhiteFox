
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(1500, 8)
        self.linear2 = torch.nn.Linear(1500, 8)
 
    def forward(self, x1, x2=None):
        v1 = self.linear1(x1)
        if x2 is not None:
            v2 = x2 * 3
            v3 = v1 + v2
        else:
            v3 = v1
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 1500)
x2 = torch.randn(1, 1500)
