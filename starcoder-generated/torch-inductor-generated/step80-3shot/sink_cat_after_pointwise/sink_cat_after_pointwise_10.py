
class SinkCat2(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = torch.cat((x, x), dim=2).tanh().view(-1, x.shape[1], 2)
        return x
# Inputs to the model
x = torch.randn(3, 2, requires_grad=True)
