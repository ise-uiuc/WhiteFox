
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        in_features, out_features = 256, 256
        self.linear = torch.nn.Linear(in_features, out_features, bias=False)
 
    def forward(self, x1, x2):
        v1 = self.linear(x1)
        v2 = v1 + x2

        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 256)
x2 = torch.randn(1, 256)
