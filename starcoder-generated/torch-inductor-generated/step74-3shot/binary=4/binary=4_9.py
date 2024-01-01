
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(in_features=30, out_features=10)
 
    def forward(self, x1, other=None):
        v1 = self.linear(x1)
        v2 = v1 + other if other is not None else v1
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 30)
x2 = torch.randn(1, 30)
