
class SinkConcat2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))
        self.conv = torch.nn.Conv2d(1, 1, 3, stride=1, padding=1)
    def forward(self, input):
        x = torch.cat((input, input), dim=0)
        x = x.shape[1]
        n, i, j = x.unbind(dim=0)
        x = torch.cat([torch.tanh(i) for i in x])
        x = x.view(n, i, j)
        return x
# Inputs to the model
x = torch.randn(2, 6, 2)
