
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.ModuleList([torch.nn.Linear(1, i * 2) for i in range(5)])
    def forward(self, x1):
        for layer in self.model:
            x1 = layer(x1)
            x1 = x1.reshape([1, -1])
            x1 = x1.reshape([1, -1])
            x1 = torch.relu(x1)
        return x1
# Inputs to the model
x1 = torch.randn(1, 1)
