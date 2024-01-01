
class Model(torch.nn.Module):
    def __init__(self, __in_features, __out_features):
        super().__init__()
        self.linear = torch.nn.Linear(__in_features, __out_features)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.sigmoid(v1)
        return v2

# Initializing the model
m = Model(16, 32)

# Inputs to the model
x1 = torch.randn(224, 224)
