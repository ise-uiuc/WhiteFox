
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
 
    def forward(self, x1):
        m1 = self.linear(x1)
        m2 = torch.nn.Sigmoid()(m1)
        m3 = m1 * m2
        return m3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
