
class my_cat(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = torch.cat((x1, x2), dim=1)
        v2 = torch.cat((x2, x1), dim=1)
        v4 = torch.cat((v1, v2), dim=1)
        v4 = torch.sigmoid(v4)
        m = v4.argmax(dim=0)
        m = m.reshape(2, -1)
        n = torch.cat((v1, m), dim=0)
        n = n.T[2:]
        return n.T
# Inputs to the model
x1 = torch.randn(2, 4, 5)
x2 = torch.randn(2, 4, 3)
