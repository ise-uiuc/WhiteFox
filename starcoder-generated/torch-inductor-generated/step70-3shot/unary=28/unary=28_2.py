
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(3, 8)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        return nn.functional.clamp(v1, min=-1e5, max=1e5)

# Initialting the model
m = Model()

# Input to the model
x1 = torch.randn(1, 3)

# Output of the model
