
class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(12, 4, 64)
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        out = self.linear(x)
        return out.relu()
# Inputs to the model
x = torch.randn(4, 3, 4)
