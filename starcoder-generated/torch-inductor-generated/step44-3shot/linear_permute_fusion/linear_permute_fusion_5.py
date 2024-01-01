
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Conv2d(2, 4, 3)
        self.conv2 = torch.nn.Conv2d(4, 8, 5)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        t1 = torch.cat((v2, v1), 1)
        w1 = torch.transpose(t1, 2, 3)
        w2 = torch.transpose(w1, 1, 2)
        w3 = w2.contiguous()
        w4 = torch.transpose(w3, 2, 3)
        w5 = torch.transpose(w4, 1, 2)
        w6 = w5.reshape((-1, 4, 16))
        return w6
# Inputs to the model
x1 = torch.randn(4, 2, 10, 10)
