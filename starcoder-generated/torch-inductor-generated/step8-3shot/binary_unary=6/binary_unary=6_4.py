
class Model(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features)
 
    def forward(self, x1, other):
        v1 = self.linear(x1)
        v2 = v1 - other
        v3 = torch.nn.functional.relu(v2)
        return v3

# Initializing the model
m = Model(5, 2)

# Inputs to the model
x1 = torch.randn(1, 5)
const = torch.randn(1, 1)
