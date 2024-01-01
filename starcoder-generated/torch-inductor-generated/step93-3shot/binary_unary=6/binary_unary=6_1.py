
class Model(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features)
 
    def forward(self, x1, x2, x3):
        v1 = self.linear(x1)
        v2 = v1 - x2
        v3 = F.relu(v2)
        return v3

# Initializing the model
m = Model(5, 6)

# Inputs to the model
x1 = torch.randn(3, 5)
x2 = torch.randn(5)
x3 = torch.randn(3, 5)
