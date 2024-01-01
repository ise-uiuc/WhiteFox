
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y1 = torch.cat((x, x), dim=1)
        y2 = torch.relu(y1)
        x = torch.tanh(y2)
        return x
# Inputs to the model
x = torch.randn(2, 3, 4)
