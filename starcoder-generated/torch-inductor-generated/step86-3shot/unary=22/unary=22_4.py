
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(224 * 224 * 3, 1000)
 
    def forward(self, x1):
        x2 = x1.view(-1, 224 * 224 * 3)
        v1 = self.fc(x2)
        v2 = torch.tanh(v1)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
