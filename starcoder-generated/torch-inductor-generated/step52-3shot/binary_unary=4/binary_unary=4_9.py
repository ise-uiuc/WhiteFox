
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(in_features=10, out_features=16)
 
    def forward(self, input_, other):
        v1 = self.linear(input_)
        v2 = v1 + other
        v3 = torch.nn.functional.relu(v2)
        return v3

# Initializing the model
m = Model()

# Creating the inputs
x = torch.randn(1, 10)
other = torch.randn(1, 16)

# Inputs to the model
