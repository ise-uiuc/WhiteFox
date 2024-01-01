
class SinkCat(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.tensor(0.0)
        self.in_features = 2
        self.out_features = 4
    def forward(self, x):
        x = torch.cat((x, x), dim=1).view(2, -1, 4)
        x = torch.relu(x)
        return x
# Inputs to the model
x = torch.randn(3, 2, requires_grad=True)
