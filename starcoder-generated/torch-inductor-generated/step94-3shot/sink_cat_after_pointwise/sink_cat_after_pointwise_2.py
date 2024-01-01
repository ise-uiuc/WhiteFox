
class SinkCat(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(4, 16)
    def forward(self, x):
        x = torch.cat((x, x), dim=0).view(2, -1)
        x = torch.tanh(x)
        return x.view(2, -1, 1)
# Inputs to the model
x = torch.randn(2, 2, 4)
