
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 32)
        self.sigmoid = torch.nn.Sigmoid()
 
    def forward(self, x):
        v1 = self.sigmoid(self.linear(x))
        v2 = v1 * x
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(32, 16)
