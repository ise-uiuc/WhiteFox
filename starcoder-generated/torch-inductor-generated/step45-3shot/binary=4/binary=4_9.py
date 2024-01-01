
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(6, 4, bias=True)

    def forward(self, x):
        v = self.linear(x)
        v = v + 1.0
        return v

# Initializing the model
m = Model()
    
# Inputs to the model
x = torch.randn(2, 6)
