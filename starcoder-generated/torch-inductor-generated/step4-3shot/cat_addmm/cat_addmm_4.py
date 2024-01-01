
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(5, 6)
        self.fc2 = torch.nn.Linear(5, 6)
 
    def forward(self, x1):
        v1 = self.fc(x1)
        v2 = self.fc2(x1)
        v3 = torch.addmm(v2, v1, v2)
        v4[:, :, 1, 1:] = v3
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 5, 128, 16)
