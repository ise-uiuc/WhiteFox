
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(32, 8)
 
    def forward(self, x1, x2):
        x = torch.cat((x1, x2), axis=1)
        v1 = self.linear(x)
        v2 = v1 - x2
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 32)
x2 = torch.randn(2, 32)
