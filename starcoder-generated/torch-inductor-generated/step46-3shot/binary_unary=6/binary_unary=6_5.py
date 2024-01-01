
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(224, 1000, bias=False)
        self.fc = torch.nn.Linear(1000, 1000)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 - 1
        v3 = torch.nn.functional.relu(v2)
        v4 = self.linear(v3)
        v5 = self.fc(v4)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 224)
