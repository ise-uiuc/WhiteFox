
class Model(torch.nn.Module):
    def __init__(self, in_features_0: int, out_features_0: int, in_features_1: int):
        super().__init__()
        self.linear = torch.nn.Linear(in_features_0, out_features_0)
        self.other = torch.nn.Parameter(torch.ones(1, in_features_1))
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + self.other
        v3 = torch.relu(v2)
        return v3
      
# Initializing the model
m = Model(11,13,17)

# Inputs to the model
x1 = torch.randn(13, 11)
