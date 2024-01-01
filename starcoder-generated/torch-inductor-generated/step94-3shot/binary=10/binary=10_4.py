
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(16, 32)
        self.linear2 = torch.nn.Linear(32, 64)
 
    def forward(self, x1):
        v1 = self.linear1(x1)
        if condition1:
            x1 = x1 + other
        v1 = self.linear2(x1)
        return v1

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 16)
