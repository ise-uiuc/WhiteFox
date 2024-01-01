
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.fc = torch.nn.Linear(4, 7)
 
    def forward(self, x1):
        v1 = self.fc(x1)
        v2 = v1 + other
        return torch.nn.ReLU()(v2)

# Initializing the model
m = Model(x1)

# Inputs to the model
