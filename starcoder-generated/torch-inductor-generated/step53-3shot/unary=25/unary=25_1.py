
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(25, 10)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 > 0
        v3 = v1 * -0.0  # Negative slope
        v4 = torch.where(v2, v1, v3)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
__input__ = torch.randn(1, 25)
