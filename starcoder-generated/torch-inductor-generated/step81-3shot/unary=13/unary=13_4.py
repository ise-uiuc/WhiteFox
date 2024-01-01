
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(120, 120)
        self.sigmoid = torch.nn.Sigmoid()
 
    def forward(self, x):
        v1 = self.linear(x)
        v2 = self.sigmoid(v1)
        v3 = v1 * v2
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 120)
