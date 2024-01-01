
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3 * 3 * 3, 16, (3, 3), (1, 1), (1, 1), 0, 1, 1, False)
    def forward(self, x1):
        v1 = self.conv1(x1)
        return v1
x1 = torch.randn(1, 3 * 3 * 3, 16, 16)
x2 = torch.randn(1, 3, 101, 101)
x3 = torch.randn(1, 3, 2, 2)
x4 = torch.randn(1, 3, 101, 101)

x5 = torch.cat((x1, x2, x4), 1)
y = torch.zeros(1, 2 * 16 + 101, 101, 101)
y[:,:,0:101,0:101] = 4 
m = Model()
# Inputs to the model
x1 = torch.randn(1, 3 * 3 * 3, 16, 16)
