
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(11, 1)
 
    def forward(self, input_tensor, other):
        v1 = self.linear(input_tensor)
        return v1 + other

# Initializing the model
m = Model()

# Inputs to the model
input_tensor = torch.randn(1, 11)
other = torch.randn(1)
