
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 12)
 
    def forward(self, x):
        v1 = torch.clamp(torch.clamp(x, min=0, max=6) + 3, min=0, max=6)
        v2 = v1 * 6
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 8)
