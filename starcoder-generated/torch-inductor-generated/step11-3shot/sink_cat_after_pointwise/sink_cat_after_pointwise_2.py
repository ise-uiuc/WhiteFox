
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        v = torch.relu(x)
        v = torch.cat((v, v, v, v), dim=1)
        v = v*v+x*x
        return v.view(-1)
# Inputs to the model
x = torch.randn(5, 2)
