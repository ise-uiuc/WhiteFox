
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        t = torch.ones(10, 20)
        y = torch.cat((t, x), dim=0)
        return y.view(y.shape[0], -1).relu()
# Inputs to the model
x = torch.ones(1, 20)
