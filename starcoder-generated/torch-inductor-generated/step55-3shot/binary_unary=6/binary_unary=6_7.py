
class Model(torch.nn.Module):
    def __init__(self, input_shape):
        super().__init__()

        self.linear = torch.nn.Linear(input_shape[0], 16)

    def forward(self, x):
        v1 = self.linear(x)
        v2 = v1 - other
        v3 = torch.nn.functional.relu(v2)
        return v3
        
# Initializing the model
m = Model((3, 6))

# Inputs to the model
x = torch.randn(1, 3, 64, 64)
