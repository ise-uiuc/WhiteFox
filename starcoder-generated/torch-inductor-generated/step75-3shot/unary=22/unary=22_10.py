
class Model(torch.nn.Module):
    def __init__(self, in_features, features_out):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, features_out, bias=False)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.tanh(v1)
        return v2

# Initializing the model
m = Model(160, 1)

# Inputs to the model
x1 = torch.randn(1, 160)
