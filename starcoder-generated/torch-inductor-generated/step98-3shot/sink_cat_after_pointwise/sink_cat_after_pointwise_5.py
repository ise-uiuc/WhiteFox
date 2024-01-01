
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = 1 + x.view(2 * x.shape[0]).relu().tanh() + torch.cat((x, x), dim=1)
        return x
# Inputs to the model
x = torch.randn(2, 3, 4)
