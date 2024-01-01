
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.concat((x, x), dim=1)
        aa = torch.reshape(y, 5, -1)
        bb = torch.tanh(aa)
        cc = torch.matmul(bb, bb.shape[0], bb.shape[0])
        return cc
# Inputs to the model
x = torch.randn(5, 3, 4)
y = torch.randn(5, 3, 4)
bb = torch.reshape(y, 15, -1)
