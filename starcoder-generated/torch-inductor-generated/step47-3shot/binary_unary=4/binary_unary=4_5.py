
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 3)
 
    def forward(self, x1, tensor):
        v1 = self.linear(x1)
        v2 = v1 + tensor
        v3 = v2.relu()
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 2)
tensor = torch.randn(1, 3)
