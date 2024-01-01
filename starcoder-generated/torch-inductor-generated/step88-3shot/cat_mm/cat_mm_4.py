
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        l1 = 15
        l2 = 5
        v0 = x[0].view(1, 1, 2, 10 * l1, 2, 5, 5, 3 * l2)
        v1 = x[1].view(10 * l1)
        return torch.cat((v0, v1))
# Inputs to the model
x = torch.randn(2, 3, 4, 5, 6, 7, 8, 9)
