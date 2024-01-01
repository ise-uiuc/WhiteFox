
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(2, 4)
        self.fc2 = torch.nn.Linear(2, 6)
    def forward(self, x, x1):
        x = self.fc1(x)
        x = torch.cat((x, x1), dim=1)
        x = self.fc2(x)
        x = torch.relu(x)
        return x
# Inputs to the model
x = torch.randn(2, 2)
x1 = torch.randn(2, 3)
