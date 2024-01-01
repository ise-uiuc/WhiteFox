
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(128, 128)
 
    def forward(self, x1, x2):
        v1 = self.linear(x1)
        v = v1 + x2
        return v

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(4, 128)
x2 = torch.randn(4, 128)
