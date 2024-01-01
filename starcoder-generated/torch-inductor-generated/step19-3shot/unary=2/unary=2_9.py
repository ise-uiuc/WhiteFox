
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(1000 * 1, 4)

    def forward(self, x1):
        v1 = x1.view((x1.shape[0], -1))
        v2 = self.fc(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
