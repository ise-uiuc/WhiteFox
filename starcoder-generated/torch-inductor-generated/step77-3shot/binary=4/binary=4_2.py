
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(128, 128)
 
    def forward(self, x1, other):
        t1 = self.fc(x1)
        t2 = t1 + other
        return t2

# Initializing the model
a = torch.tensor(1)
m = Model()

# Inputs to the model
x1 = torch.randn(1, 128)
