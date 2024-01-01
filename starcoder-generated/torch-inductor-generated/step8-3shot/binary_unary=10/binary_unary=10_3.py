
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(224 * 224 * 3, 500)
 
    def forward(self, x1):
        v1 = x1.view(-1, 224 * 224 * 3)
        v2 = self.fc(v1)
        v3 = v2 + 10
        v4 = torch.nn.functional.relu(v3)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
