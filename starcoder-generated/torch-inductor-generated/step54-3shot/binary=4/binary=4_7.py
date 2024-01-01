
class Model(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.linear = torch.nn.Linear(width, 1, bias=True)
        self.other = torch.randn(width, 1)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + self.other
        return v2

# Initializing the model
w = 50
m = Model(w)

# Inputs to the model
x1 = torch.randn(1, 1, w)
