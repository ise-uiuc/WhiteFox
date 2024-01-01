
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu6 = torch.nn.Sequential(
            torch.nn.Linear(3, 4),
            torch.nn.ReLU6(inplace=True)
        )
 
    def forward(self, x2):
        j1 = self.linear_relu6(x2)
        q1 = j1 * torch.clamp(j1 + 3, min=0, max=6)
        y = q1 / 6
        return y

# Initializing the model
m = Model()

# Inputs to the model
x2 = torch.randn(1, 3, 2)
