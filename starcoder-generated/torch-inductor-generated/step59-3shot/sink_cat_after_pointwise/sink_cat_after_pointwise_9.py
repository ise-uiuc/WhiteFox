
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = torch.cat((x, x), dim=1)
        return x if x.shape[1] > 2 else x.view(-1)
# Inputs to the model
x = torch.randn(5, 2, 3, 4)
