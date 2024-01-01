
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
 
    def forward(self, x1):
        v1 = self.linear(x1) # Linear transformation
        v2 = v1 + 3
        v3 = torch.clamp_min(v2, 0) # Clamp the output of the addition to a minimum
        v4 = torch.clamp_max(v3, 6) # Clamp the output of the previous operation to a maximum
        v5 = v4 / 6
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
