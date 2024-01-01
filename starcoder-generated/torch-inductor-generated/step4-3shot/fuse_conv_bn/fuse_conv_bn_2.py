
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(3, 30, 3)
        self.fc = torch.nn.Linear(15, 30)
    def forward(self, x):
        x = self.conv(x)
        x = x.mean(-1)        # reduce the last dimension
        x = self.fc(x)
        return x
# Inputs to the model
x = torch.randn(10, 3, 4)
