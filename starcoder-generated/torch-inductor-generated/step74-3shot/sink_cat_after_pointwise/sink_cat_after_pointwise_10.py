
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y0 = torch.cat((x, x, x), dim=1)
        y1 = torch.relu(y0)
        y2 = torch.cat((x, y1), dim=0)
        return y2.view(y2.shape[0], -1).tanh()
# Inputs to the model
x = torch.randn(2, 3, 4)
