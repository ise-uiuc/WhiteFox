
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 50)
 
    def forward(self, x2):
        v1 = self.linear(x2)
        v2 = v1 + 1  # Add a small value (to avoid being 0) for the second tensor, to ensure that we get nonzero outputs
        return v2
 
# Initializing the model
m = Model()

# Inputs to the model
x2 = torch.normal(0, 1, (1, 10))
