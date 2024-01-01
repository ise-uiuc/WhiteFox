
class Model(torch.nn.Module):
    def __init__(self, input_features, output_features):
        super().__init__()
        self.linear = torch.nn.Linear(input_features, output_features)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.sigmoid(v1)
        return v2

# Initializing the model
m = Model(5, 2)

# Inputs to the model
x1 = torch.randn(1, 5)
