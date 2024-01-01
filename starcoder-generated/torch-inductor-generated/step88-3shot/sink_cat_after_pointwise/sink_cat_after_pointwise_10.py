
class SinkTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = torch.tanh(x + torch.cat((x, x), dim=1).view(x.shape[0], -1))
        return x
# Inputs to the model
x = torch.randn(3, 2, requires_grad=True)
