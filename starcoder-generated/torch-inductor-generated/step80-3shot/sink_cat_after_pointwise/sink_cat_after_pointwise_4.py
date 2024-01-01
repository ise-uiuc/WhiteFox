
class SinkCat_ReLU(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.tensor(0.0)
        self.in_features = 2
        self.out_features = 4
    def forward(self, x):
        y =  torch.relu(torch.cat((x, x), dim=1).view(x.shape[0], -1))
        y =  torch.tanh(y) if (y.shape[0] == 1) else y.tanh()
        return y
# Inputs to the model
x = torch.randn(3, 2, requires_grad=True)
