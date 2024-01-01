
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(10, 32)
        self.fc2 = torch.nn.Linear(32, 32)
 
    def forward(self, input):
        v1 = self.fc1(input)
        v2 = self.fc2(v1)
        v3 = torch.cat([input, v2], dim=1)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
input = torch.randn(5, 10)
