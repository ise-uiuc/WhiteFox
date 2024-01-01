
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(15, 8)
 
    def forward(self, x2):
        v1 = self.fc(x2)
        v2 = v1 + torch.normal(3, 100, (8,), requires_grad=True)
        v3 = torch.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x2 = torch.randn(3, 15)
