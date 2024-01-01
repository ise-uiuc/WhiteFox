
class Model(torch.nn.Module):
    def __init__(self, linear_in_features, linear_out_features):
        super().__init__()
        self.linear = torch.nn.Linear(linear_in_features, linear_out_features)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + other
        return v2

# Initializing the model
m = Model(1, 3)

# Inputs to the model
x1 = torch.randn(1, 1)
