
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        __output_dim__ = __input_dim__ * __factor__
        self.linear = torch.nn.Linear(__input_dim__, __output_dim__)
 
    def forward(self, x):
        v1 = self.linear(x)
        v2 = v1 + __other__
        v3 = torch.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(__input_dim__)
