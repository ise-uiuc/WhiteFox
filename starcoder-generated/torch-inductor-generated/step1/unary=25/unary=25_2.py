
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 128)
 
    def forward(self, x):
        v1 = self.linear(x)
        v2 = torch.gt(v1, 0.)
        v3 = v1
        v4 = v2 * v3
        x = v4
        return x

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 10)
