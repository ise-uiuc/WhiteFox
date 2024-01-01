
class SinkCat(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.tensor(0.0)
        self.in_features = 4
        self.out_features = 7
    def forward(self, x):
        x = torch.tanh(torch.cat((x, x), dim=1).view(x.shape[0], -1))
        return x
# Inputs to the model
x = torch.randn(3, 4, requires_grad=True)
