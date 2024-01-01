
class Model(torch.nn.Module):
    def __init__(self, in_features, out_features, bias):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(in_features, out_features, bias)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp(max=6, min=0, input=v1 + 3) * v1
        v3 = v2 / 6
        return v3

# Initializing the model
m = Model(256, 2, True)

# Inputs to the model
x1 = torch.randn(1, 256),
