
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)
 
    def forward(self, x0):
        v1 = self.linear(x0)
        v2 = v1.sigmoid()
        v3 = v1 * v2
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x0 = torch.randn(2, 10)
