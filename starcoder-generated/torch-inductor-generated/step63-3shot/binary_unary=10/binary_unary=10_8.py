
class Model(torch.nn.Module):
    def __init__(self):
        self._fc1 = torch.nn.Linear(3, 16)
        self._fc2 = torch.nn.Linear(16, 2)
 
    def forward(self, inputs):
        x = inputs
        x = self._fc1(x)
        x = x + x
        x = F.relu(x)
        return self._fc2(x)

# Inititializing the model.
model = Model()

# Inputs to the model.
inputs = torch.randn(1, 3)

# Output of the model.
