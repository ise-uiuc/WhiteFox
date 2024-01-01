
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(3, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 10)
        )
 
    def forward(self, x1):
        v1 = self.mlp(x1)
        return v1

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
x2 = torch.randn(1, 10)
x3 = torch.randn(6, 1)
