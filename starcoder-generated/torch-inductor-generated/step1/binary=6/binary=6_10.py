
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Linear(5, 8, bias=False)
 
    def forward(self, x, other):
        v1 = self.conv(x)
        v6 = v1 - other
        return v6

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 5)  # the input is not important here
other = torch.randn(1, 8)
