
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 1)
 
    def forward(self, x1):
        v1 = self.dropout(x1, 0.2)
        v2 = torch.clamp(v1, min=0, max=6)
        v3 = v2 + 3
        v4 = v3 * 0.16666666666666666
        v5 = v4 / 6
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8)
