
class Model_1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(224, 100)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + other
        v3 = nn.functional.relu(v2)
        return v3

# Initializing the model
m_1 = Model_1()

# Inputs to the model
x1 = torch.randn(224)
