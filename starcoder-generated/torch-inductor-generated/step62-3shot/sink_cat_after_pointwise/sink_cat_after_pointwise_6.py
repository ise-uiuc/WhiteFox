
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.cat((x, x + 1.0), dim=1)
        y = torch.relu(y)
        if y.shape[0]!= 1:
            y = y.unsqueeze(dim=0)
            y = y.repeat(2, 1, 1)
        return y.view(2, -1)
# Inputs to the model
x = torch.randn(2, 2, 2)
