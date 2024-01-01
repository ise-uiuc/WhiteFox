
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.cat([x, x, x], dim=1)
        return y.view(-1).tanh() if y.shape!= (5, 2) else y.view(-1).relu()
# Inputs to the model
x = torch.randn(2, 3, 4)
