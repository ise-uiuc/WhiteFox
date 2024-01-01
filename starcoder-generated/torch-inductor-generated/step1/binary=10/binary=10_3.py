
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 32, bias=False)
 
    def forward(self, v, x):
        return v * self.linear(x) + self.other

# Initializing the model
m = Model()

# Other input tensor
m.other = torch.ones(1)

# Inputs to the model
v = torch.ones(1)
x = torch.ones(1, 3, 64, 64)
