
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_ = torch.nn.Linear(10, 10)
 
    def forward(self, x1):
        v1 = self.linear_(x1)
        v2 = v1 - x1
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(10)
