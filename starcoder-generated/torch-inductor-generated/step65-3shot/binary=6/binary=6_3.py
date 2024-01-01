
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
 
    def forward(self, x2):
        v7 = self.linear(x2)
        v8 = v7 - 0.123
        return v8

# Initializing the model
m = Model()

# Inputs to the model
x2 = torch.randn(1, 3)
