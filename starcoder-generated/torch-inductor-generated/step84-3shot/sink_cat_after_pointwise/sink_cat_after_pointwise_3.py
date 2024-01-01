
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))
        self.in_features = 2
        self.out_features = 4
    def forward(self, x):
        x = torch.cat([x, x], dim=1).reshape(self.out_features, -1)
        x = torch.relu(x)
        return x
# Inputs to the model
x = torch.randn(3, 2, requires_grad=True)
