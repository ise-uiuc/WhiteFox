
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 64)
 
    def forward(self, x1, x2):
        v1 = self.linear(x2)
        v2 = v1 - x1
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 512, 7, 7)
x2 = torch.randn(2, 16)
