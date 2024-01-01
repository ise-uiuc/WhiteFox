
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 8, bias=False)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 - 3.156377799505772
        v3 = torch.clamp(v2, min=1.747891573390991, max=9.727620211029053)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8)
