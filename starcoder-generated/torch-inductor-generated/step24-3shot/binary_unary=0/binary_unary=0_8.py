, note: the second relu is applied to the result of previous operation.
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 16, 1, stride=1)
    def forward(self, x):
        v = self.conv(x)
        v = torch.relu(v)
        v = torch.relu(v)
        return v
# Inputs to the model
x = torch.randn(1, 16, 64, 64)
