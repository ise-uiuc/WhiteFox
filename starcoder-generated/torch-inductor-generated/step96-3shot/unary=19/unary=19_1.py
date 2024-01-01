
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
 
    def forward(self, x):
        v = self.linear(x)
        v2 = torch.sigmoid(v)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3)
