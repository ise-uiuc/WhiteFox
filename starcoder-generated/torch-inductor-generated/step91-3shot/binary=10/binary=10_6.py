
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 11)
 
    def forward(self, v):
        v1 = self.linear(v)
        v2 = v1 + 100
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(5, 10)
