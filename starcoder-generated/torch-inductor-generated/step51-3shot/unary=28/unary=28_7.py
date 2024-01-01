
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(28 * 28, 28 * 28)
 
    def forward(self, x1, max_value=127):
        v1 = self.linear(x1)
        v2 = v1.clamp(min_=-255.0)
        return v2.clamp(max=max_value)

# Initializing the model
__m__ = Model()

# Inputs to the model
x1 = torch.randn(1, 28 * 28)
