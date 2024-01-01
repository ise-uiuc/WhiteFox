
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3 * 3, 6)

    def forward(self, input):
        x = x.permute(0, 2, 1)
        x1 = self.fc(x)
        return x1, x1
# Inputs to the model
x = torch.randn(4, 3, 3)
