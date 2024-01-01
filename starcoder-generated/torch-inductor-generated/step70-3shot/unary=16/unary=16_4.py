_1
class Model_1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(32, 8)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.relu(v1)
        return v2

# Initializing the model
m2 = Model_1()

# Inputs to the model
x1 = torch.randn(128, 32)
v3 = m2(x1)


