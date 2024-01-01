
class Model(torch.nn.Module):
    def __init__(self)
        super().__init__()
        self.linear1 = torch.nn.Linear(16, 12)
 
    def forward(self, x1, another):
        v1 = self.linear1(x1)
        v2 = v1 + another
        return v2

# Initializing the model
another = torch.randn(1, 12)
m = Model()

# Inputs to the model
x1 = torch.randn(1, 16)
