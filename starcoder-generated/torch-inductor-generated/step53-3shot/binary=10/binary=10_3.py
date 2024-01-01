
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(5, 10)
        self.other = torch.randn(1, 10)
    
    def forward(self, x):
        v1 = self.linear(x)
        v2 = v1 + self.other
        return v2


# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 5)
