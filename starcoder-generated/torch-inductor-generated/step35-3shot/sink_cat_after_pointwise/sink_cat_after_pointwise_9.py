
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    @staticmethod
    def relu(x):
        return x.clamp(0)
    def forward(self, x):
        y = torch.cat((x, x), dim=1)
        y = y.view(y.shape[0], -1).relu()
        y = y if y.shape[0] == 1 else y.relu()
        return y
# Inputs to the model
x = torch.randn(2, 3, 4)
