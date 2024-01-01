
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(3, 5, bias=False)
 
    def forward(self, x1, x2):
        v1 = torch.relu(x2 - self.fc(x1))
        return v1

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(3, 4, 2)
x2 = torch.randn(3, 4, 2)
