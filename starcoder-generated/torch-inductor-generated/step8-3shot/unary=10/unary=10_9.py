
class Model(torch.nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.linear = torch.nn.Linear(num_features, 64)
 
    def forward(self, x):
        v1 = self.linear(x)
        v2 = v1 + 3
        v3 = F.relu6(v2, inplace=True)
        return v3 / 6

# Initializing the model
m = Model(64)

# Inputs to the model
x = torch.randn(1, 64)
