
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        t = torch.cat((x, x), dim=1)
        x = torch.sigmoid(t.view(t.size()[0], -1)).view(-1, t.size()[1], t.size()[2])
        return x
# Inputs to the model
x = torch.randn(2, 3, 4)
