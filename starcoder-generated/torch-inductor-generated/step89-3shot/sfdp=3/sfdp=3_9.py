
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(50, 50)
        self.fc2 = torch.nn.Linear(50, 50)
        self.fc3 = torch.nn.Linear(50, 50)
 
    def forward(self, x1, x2, x3):
        t1 = torch.cat([x1, x2, x3], 1)
        t2 = torch.tanh(self.fc1(t1))
        t3 = torch.tanh(self.fc2(t2))
        t4 = torch.tanh(self.fc3(t3))
        return t4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 50)
x2 = torch.randn(1, 50)
x3 = torch.randn(1, 50)
