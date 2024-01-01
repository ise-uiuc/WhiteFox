
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 2)
 
    def forward(self, x):
        v1 = self.linear(x)
        v2 = v1.sigmoid()
        v3 = v2 * v1
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(12, 8)
