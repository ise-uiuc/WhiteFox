
class CatTanhReshapeReLU(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.in_features = 3
        self.out_features = 4
    def forward(self, x):
        y = torch.tanh(torch.cat((x, x, x), dim=1).view(-1, 3, 2, 1))
        if not y.shape == (6, 3, 2, 1):
            y = torch.relu(y)
        return y.reshape(y.size(0), y.size(2), y.size(3))
# Inputs to the model
x = torch.randn(3, 2, requires_grad=True)
