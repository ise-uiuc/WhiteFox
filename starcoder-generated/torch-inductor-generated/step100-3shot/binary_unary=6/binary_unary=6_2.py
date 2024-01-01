
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 4, bias=True)
 
    def forward(self, x2):
        v2 = x2
        v3 = self.linear(v2)
        v4 = v3 - 7
        v5 = v4.clamp(min=0, max=7)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x2 = torch.randn(1, 8)
