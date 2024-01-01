
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(18, 10)
 
    def forward(self, input, other):
        v1 = self.linear(input)
        v2 = v1 + other
        v3 = torch.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
__input_tensor__ = torch.randn(3, 18)
other = torch.randn(3, 10)
