
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = x.repeat(x.shape[-2], x.shape[-1])
        z = torch.cat((y, y), dim=1)
        x = z.view(z.shape[0], -1).tanh()
        return x
# Inputs to the model
x = torch.randn(2, 4, 5)
