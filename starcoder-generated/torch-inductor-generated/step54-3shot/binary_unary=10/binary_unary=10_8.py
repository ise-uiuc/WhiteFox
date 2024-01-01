
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(123, 321)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        # Fill in the missing line to generate the final output tensor.
        v2 = v1 + torch.randn(321)
        v3 = torch.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 123)
