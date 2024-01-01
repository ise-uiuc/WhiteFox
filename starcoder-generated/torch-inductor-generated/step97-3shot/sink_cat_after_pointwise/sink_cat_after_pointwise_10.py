
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.cat([x.clone(), x], dim=0)
        y = y.relu().view(4, 4)
        y = y.tanh().clone()
        return y
# Inputs to the model
x = torch.randn(2, 3, 4)
