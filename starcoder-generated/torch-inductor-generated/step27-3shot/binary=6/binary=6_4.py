
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(4, 2)
 
    def forward(self, x1, x2):
        m1 = self.fc(x1)
        m2 = m1 - x2
        return m2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(10,4)
x2 = 2
