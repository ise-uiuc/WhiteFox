
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        w = x.view(-1).relu().view(1, -1)
        x = w.expand(-1, 3)
        y = x.squeeze(0)
        return y
# Inputs to the model
x = torch.randn(2, 1, 1).expand(1, 3, 1)
