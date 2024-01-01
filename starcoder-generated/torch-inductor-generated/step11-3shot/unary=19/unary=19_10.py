
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(81, 10)
 
    def forward(self, x1):
        x2 = x1.flatten(-2, -1)
        v1 = self.linear(x2)
        v2 = torch.sigmoid(v1)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 7, 7)
