
class Layer(torch.nn.Module):
    def __init__(self, output_size, bias):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, output_size, bias)
 
    def forward(self, x):
        v1 = self.linear(x)
        v2 = v1 + other
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 5)
