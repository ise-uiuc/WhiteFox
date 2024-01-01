
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1 + 2, 2)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.tanh(v1)
        return v2

# Initializing the model
m = Model()

# Generate random input to the generated model.
input_tensor = torch.randn(1, 1 + 2)
