
class SinkCat(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = torch.cat((x, torch.nn.ReLU(x)), dim=1).view(x.shape[0], -1)
        x = torch.tanh(x)
        return x
# Inputs to the model
x = torch.randn(3, 2, requires_grad=True)
