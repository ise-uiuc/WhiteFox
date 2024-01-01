
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 2)
 
    def forward(self, x3):
        v3 = self.linear(x3)
        v2 = torch.sigmoid(v3)
        v6 = v3 * v2
        return v6

# Initializing the model
m = Model()

# Inputs to the model
x3 = torch.randn(1, 4)
