
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.c1 = nn.ConvTranspose3d(19, 48, (8, 6, 9), 1, (3, 2, 4), 1, 2)
    def forward(self, x11):
        n1 = self.c1(x11)
        n2 = n1 > 0
        n3 = n1 * -0.201
        n4 = torch.where(n2, n1, n3)
        return torch.nn.functional.adaptive_avg_pool3d(torch.nn.functional.relu(n4), 4)
# Inputs to the model
x11 = torch.randn(85, 19, 17, 8, 3)
