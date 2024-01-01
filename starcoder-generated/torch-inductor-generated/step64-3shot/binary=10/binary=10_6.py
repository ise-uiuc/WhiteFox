
class Model(torch.nn.Module):
    def __init__(self, in_features, out_features, bias):
        super().__init__()
        self.fc = torch.nn.Linear(in_features, out_features, bias)
 
    def forward(self, x1, other):
        v1 = self.fc(x1)
        v2 = v1 + other
        return v2

# Initializing the model
m = Model(3, 8, True)

# Inputs to the model
x1 = torch.randn(1, 3)
other = torch.randn(1, 8)
