
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(4, 5)
 
    def forward(self, x1):
        v1 = self.layer(x1)
        return v1

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 4)
other = torch.randn(1, 5)
