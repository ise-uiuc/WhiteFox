
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu = torch.nn.Sequential(
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(True)
        )
 
    def forward(self, x1):
        v1 = self.linear_relu(x1)
        return v1

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 64)
