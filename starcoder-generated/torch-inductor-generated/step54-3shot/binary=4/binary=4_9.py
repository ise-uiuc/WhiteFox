
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(10, 8)
 
    def forward(self, x1):
        x2 = torch.ones([1, 1, 1, 8])
        y = torch.add(self.fc(x1), x2)
        return y

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 10)
