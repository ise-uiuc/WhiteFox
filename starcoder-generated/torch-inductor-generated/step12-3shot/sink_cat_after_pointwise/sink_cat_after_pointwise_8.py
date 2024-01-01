
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = x.view(x.shape[0], -1)
        y.tanh()
        z = torch.cat((y, y), dim=1)
        x = z.view(z.shape[0], -1).tanh() if torch.numel(z) == 1 else z.tanh()
        return x
# Inputs to the model
x = torch.randn(2, 3, 4)
