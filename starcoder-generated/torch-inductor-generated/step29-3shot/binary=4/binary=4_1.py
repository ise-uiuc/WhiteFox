
class Model(torch.nn.Module):
    def __init__(self, w):
        super().__init__()
        self.linear = torch.nn.Linear(100, 100)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + w
        return v2

# Initializing the model
w = torch.randn(100)
m = Model(w)

# Inputs to the model
x1 = torch.randn(100)
