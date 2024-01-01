
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(128,1)
 
    def forward(self, x1):
        return torch.sigmoid(self.fc(x1))

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(10, 128)
