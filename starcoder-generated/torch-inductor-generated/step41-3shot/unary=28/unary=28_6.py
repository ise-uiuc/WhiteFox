
class Model(torch.nn.Module):
    def __init__(self, min_value, max_value):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features)
        self.min_value = min_value
        self.max_value = max_value
 
    def forward(self, x):
        v1 = self.linear(x)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        return v3

# Initializing the model
min_value = 0.2
max_value = 0.4
m = Model(min_value=min_value, max_value=max_value)

# Input to the model
x = torch.randn(n, n)
