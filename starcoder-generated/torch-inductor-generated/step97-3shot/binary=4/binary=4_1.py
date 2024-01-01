
class Model(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features)
 
    def forward(self, x1, other):
        v1 = self.linear(x1)
        v2 = v1 + other
        return v2

# Initializing the model
m = Model(10, 20)

# Inputs to the model
x1 = torch.randn(1, 10)
other = torch.randn(1, 20)
