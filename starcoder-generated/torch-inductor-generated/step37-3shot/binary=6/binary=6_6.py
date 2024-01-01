 
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        in_features = 10
        out_features = 20
        self.linear = torch.nn.Linear(in_features, out_features, False)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 - 20
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 10)
