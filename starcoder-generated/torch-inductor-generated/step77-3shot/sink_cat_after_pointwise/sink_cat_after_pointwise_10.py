
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.cat((x, x, x, x), dim=1)
        z = x.view(x.shape[0], -1) # Concatenated tensor not used
        return z.tanh()
# Inputs to the model
x = torch.randn(2, 3, 4)
