
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, x3, x4, x5, x6, x7, x8, x9):
        qk = torch.matmul(x1, x2.transpose(-2, -1))
        scaled_qk = qk.div(x3)
        qk = self._softmax(scaled_qk, x4)
        q = torch.nn.functional.dropout(qk, x5)
        o = torch.matmul(q, x6)
        return o

    def _softmax(self, x, dim_size):
        xn = F.softmax(x, dim=-1)
        return xn
m = Model()

# Inputs to the model
x1 = torch.randn(12, 64, 512)
x2 = torch.randn(12, 64, 64)
x3 = torch.randn(64)
