
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features)
 
        self.min_value = torch.randn(1, in_features) 
        self.max_value = torch.randn(1, in_features)
 
    def forward(self, x):
        y = self.linear(x)
        z = torch.clamp_min(y, self.min_value)
        return torch.clamp_max(z, self.max_value)

# Initializing the model
m = Model()
in_features = 4
out_features = 8
