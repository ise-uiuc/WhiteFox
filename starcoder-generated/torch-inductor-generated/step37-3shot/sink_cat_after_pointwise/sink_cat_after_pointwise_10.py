
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = x.view(-1)
        x = x.view(x.shape[0], 11, 11)
        x = x + x
        x = torch.sum(x, dim=(2, 3), keepdim=True)
        return x
# Inputs to the model
x = torch.randn(1, 15, 25, 50)
