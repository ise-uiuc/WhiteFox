
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 8)
 
    def forward(self, x3):
        v3 = self.linear(x3)
        v4 = torch.sigmoid(v3)
        v5 = v3 * v4
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x3 = torch.randn(1, 2)
