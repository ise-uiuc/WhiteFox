
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(8, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
        )
 
    def forward(self, x1):
        v1 = self.linear_relu_stack(x1)
        return v1

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(3, 8)
x2 = torch.randn(7, 6)
