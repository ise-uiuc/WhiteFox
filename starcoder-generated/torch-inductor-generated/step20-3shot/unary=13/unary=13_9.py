
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
 
    def forward(self, x):
        v = self.linear(x)
        v1 = torch.sigmoid(v)
        v2 = v * v1
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(2, 3)
