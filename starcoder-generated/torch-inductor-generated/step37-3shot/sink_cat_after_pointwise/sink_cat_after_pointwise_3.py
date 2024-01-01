
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = torch.cat((x, x), dim=1)
        x = x.view(x.shape[0], -1).pow(2)
        x = torch.relu(x)
        x = x.view(x.shape[0], -1)
        x = torch.sum(x.sin(), dim=1, keepdim=True)
        x = torch.cos(x)
        return x
# Inputs to the model
x = torch.randn(2, 3, 4)
