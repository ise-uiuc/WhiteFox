
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.module = torch.nn.Linear(5, 5)
 
    def forward(self, x):
        x = self.module(x)
        x = x - 0.9  # The value to be subtracted from the output of the linear transformation
        x = F.relu(x)
        return x

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 5)
