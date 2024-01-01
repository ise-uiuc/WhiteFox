
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(20, 20)
    def forward(self, x):
        x = torch.cat((x, x), dim=1)
        x = self.fc(x)
        return x
# Inputs to the model
x = torch.randn(5, 20)
