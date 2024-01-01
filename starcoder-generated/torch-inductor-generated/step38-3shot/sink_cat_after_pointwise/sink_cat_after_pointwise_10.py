
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = torch.cat((x.view(x.shape[0], -1).relu(), x.view(x.shape[0], -1).relu()), dim=1) if x.shape[1]!= 2 else torch.cat((x.view(x.shape[0], -1).relu(), x), dim=1)
        x = x.view(x.shape[0], -1).tanh()
        return x
# Inputs to the model
x = torch.randn(2, 2, 2)
