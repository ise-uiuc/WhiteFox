
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        a = torch.cat((x, x), dim=0)
        x = a.view(a.shape[0], -1)
        y = torch.relu(x) if x.shape[0] == 3 else torch.tanh(x)
        x = y.view(x.shape[0], int(np.sqrt(x.shape[1])), int(np.sqrt(x.shape[1])))
        return x
# Inputs to the model
x = torch.randn(3, 4, 5)
