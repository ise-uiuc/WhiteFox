
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(64, 10)
 
    def forward(self, x1, other=torch.randn(1, 10)):
        v1 = self.linear(x1)
        v2 = v1 + other
        return F.relu(v2)

# Initialize the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 64)
