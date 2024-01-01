
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(20, 10, bias=False)
        self.nonlinear = torch.nn.ReLU()
 
    def forward(self, x1):
        v1 = self.fc(x1)
        v2 = self.nonlinear(v1)
        return v2

# Initializing the model
m = Model()

# Input to the model
x1 = torch.randn(1, 20)
