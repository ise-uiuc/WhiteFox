
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.Conv_2 = nn.Conv2d(3, 2, 2, stride=1)
        self.Conv_1 = nn.Conv2d(2, 2, 2, stride=2)
        self.Conv_3 = nn.Conv2d(2, 1, 2, stride=1)
    def forward(self, x_t):
        t2 = self.Conv_1(self.Conv_2(x_t))
        t1 = self.Conv_3(t2)
        t3 =torch.sigmoid(t1)
        return t3
# Inputs to the model
x1 = torch.randn(1, 3, 32, 64)
