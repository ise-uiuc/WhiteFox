
def make_model(other):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(3, 4)
 
        def forward(self, x1):
            v1 = self.linear(x1)
            v2 = v1 + other
            v3 = torch.relu(v2)
            return v3
    return Model

# Initializing the model
m = make_model(other=10)

# Create the inputs
x1 = torch.randn(1, 3, 64, 64)

# Run the model
