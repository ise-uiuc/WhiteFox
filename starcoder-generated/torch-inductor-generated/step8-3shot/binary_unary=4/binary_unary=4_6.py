
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(5, 10)
 
    def forward(self, __input_tensor, *, other):
        v1 = self.linear(__input_tensor)
        v2 = v1 + other
        v3 = torch.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 5)
other = torch.rand(1, 10)
