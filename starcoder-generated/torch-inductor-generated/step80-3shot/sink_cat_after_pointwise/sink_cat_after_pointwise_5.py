
class SinkCatTwoInputs(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.tensor(0.0)
        self.in_features = 2
        self.out_features = 4
    def forward(self, x, y):
        x = torch.cat((x, x), dim=1).view(x.shape[0], -1)
        x = torch.relu(x)
        x = torch.sum(x * y)
        return x
# Inputs to the model
x = torch.randn(3, 2, requires_grad=True)
y = torch.randn(3, 2, requires_grad=True)
