
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y1 = torch.reshape(torch.cat([x, x], dim=0), (x.shape[0], -1))
        y2 = y1.tanh()
        return y2
# Inputs to the model
x = torch.randn(2, 3, 4)
