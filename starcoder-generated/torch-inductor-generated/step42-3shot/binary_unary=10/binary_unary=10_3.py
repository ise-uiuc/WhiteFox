
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 5)
 
    def forward(self, x1, x2):
        v1 = self.linear(x)
        v2 = v1 + x2
        v3 = relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 4)
x2 = torch.randn(1, 5)
