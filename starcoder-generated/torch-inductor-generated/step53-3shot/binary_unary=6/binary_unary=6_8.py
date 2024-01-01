
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 16)
 
    def forward(self, x1):
        x1 = self.linear(x1)
        x1 = x1 - 1
        return torch.nn.functional.relu(x1)


# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 16)
