
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y1 = torch.cat((x, x), dim=1).view(2, 6)
        y2 = torch.cat((y1, x), dim=1)
        y3 = torch.relu(y2)
        y4 = torch.cat((y3, x), dim=1).tanh()
        return y4
# Inputs to the model
x = torch.randn(1, 3, 2)
