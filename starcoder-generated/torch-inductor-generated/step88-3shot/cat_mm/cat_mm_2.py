
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(4,40)
        self.fc2 = torch.nn.Linear(40,1)
    def forward(self, x):
        return F.relu(self.fc1(x), inplace=True)
        return self.fc2(F.relu(self.fc1(x), inplace=False))
# Inputs to the model
x = torch.randn(1, 4)
