
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.cat((x, x), dim=1)
        y = y.view(y.shape[0], -1)
        y = torch.relu(y) # <---- Here's the second user of the tensor
        return y
# Inputs to the model
x = torch.randn(2, 2, 2)
