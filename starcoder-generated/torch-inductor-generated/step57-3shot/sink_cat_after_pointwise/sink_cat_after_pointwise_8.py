
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        t1 = torch.cat((x, x), dim=3)
        t2 = t1.view(x.shape[0], -1)
        t3 = torch.relu(t2)
        z = t3 + t3
        z = torch.relu(z)
        return z
# Inputs to the model
x = torch.randn(2, 5, 3, 6)
