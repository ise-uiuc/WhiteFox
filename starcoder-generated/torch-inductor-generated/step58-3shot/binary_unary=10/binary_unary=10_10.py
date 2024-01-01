
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(64*64, 256)
 
    def forward(self, x2):
        v1 = self.fc1(x2)
        v2 = v1 + x2
        v3 = torch.nn.functional.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x2 = torch.randn(1, 64*64)
