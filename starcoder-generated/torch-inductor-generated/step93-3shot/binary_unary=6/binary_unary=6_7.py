
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(14, 8)
 
    def forward(self, x1):
        v1 = self.fc(x1)
        v2 = v1 - 0.013579889472691894
        v3 = F.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 14)
