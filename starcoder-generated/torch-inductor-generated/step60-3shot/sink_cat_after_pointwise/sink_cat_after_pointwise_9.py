
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.m = torch.nn.ReLU()
    def forward(self, x):
        return self.m(torch.cat((x, x, x), dim=0)).view(x.shape[0], -1)
# Inputs to the model
x = torch.rand(2, 3, 4)
