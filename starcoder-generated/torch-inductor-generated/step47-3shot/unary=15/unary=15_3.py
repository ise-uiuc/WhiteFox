
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc0 = torch.nn.Linear(512, 128)
        self.fc1 = torch.nn.Linear(128, 10)
    def forward(self, x1):
        x1 = x1.view((1,-1))
        x2 = self.fc0(x1)
        x3 = torch.relu(x2)
        return self.fc1(x3)
# Inputs to the model
x1 = torch.rand(1,3,28,28)
