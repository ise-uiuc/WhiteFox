
class Model(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.f1 = torch.nn.Linear(in_features, out_features)
 
    def forward(self, x1):
        v1 = self.f1(x1)
        v2 = v1 - 5
        v3 = torch.nn.functional.relu(v2)
        return v3

# Initializing the model
m = Model(5, 7)

# Inputs to the model
x1 = torch.randn(2, 5)
