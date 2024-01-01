
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(.in_features,.out_features)
 
    def forward(self, x1):
        h1 = self.linear(x1)
        h2 = h1 * torch.clamp(h1 + 3, min=0, max=6)
        h3 = h2 / 6
        return h3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1,.in_features)
