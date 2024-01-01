
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = torch.cat((x, x), dim=2)
        x = x.view(x.shape[0], -1)
        x = x.log()
        return x.reshape(x.size()[0], -1)
# Inputs to the model
x = torch.randn(2, 3, 4)
