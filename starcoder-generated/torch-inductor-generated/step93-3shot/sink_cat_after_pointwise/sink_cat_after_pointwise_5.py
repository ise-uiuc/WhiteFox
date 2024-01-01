
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = x.squeeze()
        y = torch.cat((x, x), dim=0)
        y = torch.relu(y)
        y = y.view(-1)
        x = y.view(-1)
        return x
# Inputs to the model
x = torch.randn(2, 3, 4)
