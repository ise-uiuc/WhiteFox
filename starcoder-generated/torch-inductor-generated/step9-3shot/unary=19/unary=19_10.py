
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(512, 128)
        self.fc2 = torch.nn.Linear(128, 1)
 
    def forward(self, x1):
        v1 = self.fc2(F.relu(self.fc1(x1)))
        return torch.sigmoid(v1)

# Inputs to the model
x1 = torch.randn(1, 512)
