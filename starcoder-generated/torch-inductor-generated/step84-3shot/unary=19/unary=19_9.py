
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(16,8)
 
    def forward(self, x24):
        v17 = torch.sigmoid(self.fc(x24))
        return v17

# Initializing the model
m = Model()

# Inputs to the model
x24 = torch.randn(1, 16)
