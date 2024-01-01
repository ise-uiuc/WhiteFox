
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        aaa = x.view(x.shape[0], -1)
        x = torch.cat([aaa, aaa], dim=1)
        x = torch.relu(x)
        return x
# Inputs to the model
x = torch.randn(2, 3, 4)
