
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features, bias=bias)
 
    def forward(self, x1, min_value=-1.5, max_value=1.8):
        v1 = self.linear(x1)
        v2 = torch.clamp(v1, min_value)
        v3 = torch.clamp(v2, max_value)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
in_features = 16
out_features = 32
bias = False
x1 = torch.randn(1, in_features)
