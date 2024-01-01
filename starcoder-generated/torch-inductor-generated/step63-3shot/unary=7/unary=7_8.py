
class Model(torch.nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.features = torch.nn.modules.Linear(num_features, num_features)
 
    def forward(self, x):
        y = self.features(x)
        z = y * torch.clamp(y + 3, max=6)
        w = z / 6
        return w

# Initializing the model
m = Model(num_features=128)

# Inputs to the model
x = torch.randn(1, 128)
