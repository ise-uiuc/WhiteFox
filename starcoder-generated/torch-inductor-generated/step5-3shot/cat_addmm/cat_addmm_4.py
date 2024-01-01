
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(32, 64)
        self.fc2 = torch.nn.Linear(32, 64)
 
    def forward(self, x1):
        v1 = self.fc1(x1)
        v2 = self.fc2(x1)
        v3 = torch.addmm(v1, v2, v2.transpose(0,1))
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(32, 32)
