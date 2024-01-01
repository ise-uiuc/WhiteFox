
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        aaa = torch.cat([x, x], dim=1)
        aaa = aaa.view(aaa.shape[0], -1)
        return torch.softmax(aaa)
# Inputs to the model
x = torch.randn(2, 2, 2)
