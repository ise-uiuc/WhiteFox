
class Model(torch.nn.Module):
    def __init__(self, k=1):
        super().__init__()
        self.linear = torch.nn.Linear(8, 8, True)
 
    def forward(self, x1, x2):
        v1 = self.linear(x1)
        v2 = v1 + x2
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(4, 8)
x2 = torch.randn(4, 8)
