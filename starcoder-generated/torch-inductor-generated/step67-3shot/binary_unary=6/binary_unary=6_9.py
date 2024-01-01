
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(36, 25)
 
    def forward(self, x1):
        linear_output = self.linear(x1)
        subtracted_output = linear_output - 0.5
        return torch.nn.functional.relu(subtracted_output, inplace=False)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 36)
