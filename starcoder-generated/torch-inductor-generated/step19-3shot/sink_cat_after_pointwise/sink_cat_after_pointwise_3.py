
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        t1 = torch.cat((x, x), dim=3)
        t2 = t1.view(x.shape[0], -1)
        t3 = torch.relu(t2)
        return t3
# Inputs to the model
x = torch.randn(2, 5, 4, 3, 4)
