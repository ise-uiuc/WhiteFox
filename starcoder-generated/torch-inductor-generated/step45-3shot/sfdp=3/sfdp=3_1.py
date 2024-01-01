
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v):
        qk = torch.matmul(q, k.transpose(-2, -1)) * self.scale_factor
        softmax_qk = qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(v)
        return output

m = Model()
x1 = torch.randn(1, 80, 23, 34)
x2 = torch.randn(1, 5, 23, 34)
x3 = torch.randn(1, 5, 23, 34)
