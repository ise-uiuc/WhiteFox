
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.tanh(x)
        y = torch.cat((y, y), dim=1)
        y1 = torch.relu(y)
        return y1
# Inputs to the model
x = torch.randn(2, 2, 2)
