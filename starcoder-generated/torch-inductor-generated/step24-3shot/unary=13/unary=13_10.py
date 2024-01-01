
class Model(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features)
        self.sigmoid = torch.nn.Sigmoid()
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = self.sigmoid(v1)
        v3 = v1 * v2
        return v3

# Initializing the model
m = Model(128, 64)

# Inputs to the model
x1 = torch.randn(1, 128)
