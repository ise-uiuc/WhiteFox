
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, y):
        x = torch.cat((x, x), dim=1)
        return y.view(y.shape[0], -1).relu() if y.shape!= (1, 3) else y.view(y.shape[0], -1).tanh() if x.shape!= (1, 4) else x
# Inputs to the model
x = torch.randn(2, 3, 4)
y = torch.randn(1, 3)
