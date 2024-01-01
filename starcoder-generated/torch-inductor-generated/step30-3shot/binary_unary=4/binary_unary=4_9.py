
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8, bias=True)
 
    def forward(self, x1):
        return self.linear(x1) + x1[2]

# Initializing the model
m = Model()

# Inputs to the model
other = torch.randn(8)
x1 = torch.randn(3)
