
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(4, 3)
    def forward(self, x):
        y = torch.cat((x, x, x), dim=1)
        y = y.view(y.shape[0], -1)
        return torch.relu(self.fc(y.tanh()))
# Inputs to the model
x = torch.randn(2, 4)
