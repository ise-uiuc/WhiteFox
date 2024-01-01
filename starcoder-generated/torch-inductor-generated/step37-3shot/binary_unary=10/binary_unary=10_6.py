
class Model(torch.nn.Module):
    def __init__(self, in_features=3, out_features=1, bias=True):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features, bias)
 
    def forward(self, x):
        v1 = self.linear(x)
        v2 = v1 + 1
        v3 = torch.relu(v2)
        return v3

# Initializing the model
m = Model(in_features=3, out_features=1, bias=True)

# Inputs to the model
x = torch.randn(1, 3)
