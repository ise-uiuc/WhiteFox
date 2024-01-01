
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = (torch.nn.Linear(1, 1)
                       + torch.nn.ReLU6())
                       / 6
 
    def forward(self, x1):
        v1 = self.block1(x1)
        return v1

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 1)
x2 = torch.randn(10, 1)
