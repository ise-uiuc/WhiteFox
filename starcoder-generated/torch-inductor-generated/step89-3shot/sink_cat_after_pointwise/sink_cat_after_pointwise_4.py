
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        inp = torch.relu(x.view(x.shape[0], -1))
        return inp.sigmoid()
# Inputs to the model
x = torch.randn(2, 2, 2)
