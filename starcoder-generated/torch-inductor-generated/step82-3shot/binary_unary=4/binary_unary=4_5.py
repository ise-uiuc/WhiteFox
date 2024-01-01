
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
 
    def forward(self, x1, **kwargs):
        v1 = self.linear(x1)
        v2 = v1 + 1
        return torch.nn.functional.relu(v2, inplace=True)

# Initializing the model
m = Model()

# Inputs to the model
other = torch.randn(1, 8)
