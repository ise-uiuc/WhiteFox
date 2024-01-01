
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(3, 8)
 
    def forward(self, x1):
        m1 = self.linear1(x1)
        m2 = torch.relu(m1)
        return m2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64)
