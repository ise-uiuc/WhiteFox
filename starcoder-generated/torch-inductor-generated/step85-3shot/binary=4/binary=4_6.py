
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        other = torch.zeros(20, 20)
        self.linear = torch.nn.Linear(20, 20)
 
    def forward(self, x1, x2):
        v1 = self.linear(x1)
        v2 = v1 + x2
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 20)
x2 = torch.randn(1, 20)
