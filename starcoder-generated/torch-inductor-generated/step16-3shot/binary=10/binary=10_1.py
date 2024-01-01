
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 16)
 
    def forward(self, feature_map, other):
        v1 = self.linear(feature_map)
        v2 = v1 + other
        return v2

# Initializing the model
m = Model()

# Inputs to the model
feature_map = torch.randn(2, 16)
__other__ = torch.randn(2, 16)
