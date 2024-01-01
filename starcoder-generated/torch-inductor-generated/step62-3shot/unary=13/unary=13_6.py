
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(7500, 3000)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1.sigmoid()
        v3 = v1 * v2
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(10, 7500)
