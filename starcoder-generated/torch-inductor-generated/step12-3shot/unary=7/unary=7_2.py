
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(144, 512)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 * torch.max(torch.min(v1 + 3, 6), 0)
        v3 = v2 / 6
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 144)
