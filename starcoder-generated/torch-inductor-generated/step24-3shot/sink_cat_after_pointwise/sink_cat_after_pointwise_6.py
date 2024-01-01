
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 1, bias=True)
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        x = torch.relu(x)
        return x
# Inputs to the model
x = torch.randn(2, 3, 4)
