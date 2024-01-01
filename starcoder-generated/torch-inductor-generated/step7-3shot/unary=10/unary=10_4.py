
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
 
    def forward(self, x):
        v = self.linear(x)
        v = v + 3
        v = torch.clamp_min(v, 0)
        v = torch.clamp_max(v, 6)
        v = v / 6
        return v


# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3)
# The output should be [[0.25   0.4503 0.625  0.5    0.75   0.75   0.5    0.5625]]
