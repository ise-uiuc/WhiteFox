
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(100, 128)
 
    def forward(self, x1, other):
        v1 = self.linear(x1)
        v2 = v1 + other
        v3 = torch.relu(v2)
        return v3

# Initializing the model
dummy_input = torch.randn(1, 100)

# Inputs to the model
x1 = torch.randn(1, 100)

other = torch.randn(1, 128)
