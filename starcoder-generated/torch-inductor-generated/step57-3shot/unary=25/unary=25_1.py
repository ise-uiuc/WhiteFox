
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(12, 4)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        b1 = v1 > 0
        v2 = v1 * 0.01
        v3 = torch.where(b1, v1, v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 12)
