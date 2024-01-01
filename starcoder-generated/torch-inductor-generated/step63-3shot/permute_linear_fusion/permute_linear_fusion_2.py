
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.conv2d = torch.nn.Conv2d(1, 1, (1, 1))
    def forward(self, x1):
        v1 = torch.reshape(x1, (1, 2, 2))
        v2 = self.conv2d(v1)
        return v2
# Inputs to the model
x1 = torch.randn(2, 2)
