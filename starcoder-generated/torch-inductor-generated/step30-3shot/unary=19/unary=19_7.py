
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.functional.linear
    
    def forward(self, x1):
        return self.linear(x1, torch.randn(1, 4, 1, 1))

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 24, 24)
