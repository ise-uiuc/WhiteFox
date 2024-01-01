
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(5, 5)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        u1 = F.relu(v1)
        return u1

# Initialize the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 5, 32, 32)
