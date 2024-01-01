
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        aaa = torch.cat((x, x), dim=1)
        aaa = aaa.view(aaa.shape[0], -1)
        y = torch.relu(aaa).tanh()
        return y
# Inputs to the model
x = torch.randn(5, 3, 4)
