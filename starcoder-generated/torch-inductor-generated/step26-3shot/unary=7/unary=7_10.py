
class Model(torch.nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, 1)
 
    def forward(self, x1):
         v1 = self.linear(x1)
         v2 = v1 * torch.clamp(torch.add(v1, 3), 0, 6)
         v3 = v2 / 6
         return v3

# Initializing the model
m = Model(32)

# Inputs to the model
x1 = torch.randn(10, 32)
