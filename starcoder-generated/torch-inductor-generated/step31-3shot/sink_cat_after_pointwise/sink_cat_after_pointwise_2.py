
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = torch.sigmoid(x)
        x = x.view(-1, x.shape[-1])
        y = x * x
        x = x.tanh()
        x = y * x * x
        x = torch.sum(x)
        return x
# Inputs to the model
x = torch.randn(2, 2, 2)
