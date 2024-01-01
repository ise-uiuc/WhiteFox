
class Model(torch.nn.Module):
    def __init__(self, input_dim=200, output_dim=672):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
 
    def forward(self, x):
        v1 = self.linear(x)
        v2 = v1 - x
        return v2

# Initializing the model
model = Model()

# Inputs to the model
x = torch.randn(2, model.linear.in_features)
output = model(x)

