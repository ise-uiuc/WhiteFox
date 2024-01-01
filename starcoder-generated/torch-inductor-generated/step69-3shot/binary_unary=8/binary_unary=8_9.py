
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d((1, 1, 1), 3, (1, 1), stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 + torch.ones(1,3,3,3, dtype=torch.float)
        return v2
# Inputs to the model
x1 = torch.randn(1, 1, 10, 10)
