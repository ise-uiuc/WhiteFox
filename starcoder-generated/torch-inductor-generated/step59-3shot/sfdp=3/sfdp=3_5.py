
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x3, x3, x4):
        out1 = torch.matmul(x1, x2.transpose(-2, -1))
        out2 = out1.mul(0.1)
        out3 = torch.nn.functional.softmax(out2, dim=-1)
        out4 = torch.nn.functional.dropout(out3, p=0.2)
        out5 = torch.matmul(out4, x3)
        