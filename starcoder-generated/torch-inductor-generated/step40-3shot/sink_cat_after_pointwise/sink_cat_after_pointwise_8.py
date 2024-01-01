
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.cat([x, x, x], dim=1)
        return torch.nn.functional.tanh(y.view(y.shape[0], -1)).relu() if y.shape!= (1, 3) else torch.nn.functional.sigmoid(y.view(y.shape[0], -1)).sigmoid()
# Inputs to the model
x = torch.randn(2, 3, 4)
