
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, x3, x4):
        q = x1
        k = x2
        v = x3
        s = x4
        c = q.size(-1)
        kq = torch.matmul(q, k.transpose(-2, -1))
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.div(s)
        softmax = scaled_qk.softmax(dim=-1)
        dropout = torch.nn.functional.dropout(softmax, p=0.009)
        output = dropout.matmul(v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(2, 512, 80)
k = torch.randn(2, 512, 80)
v = torch.randn(2, 512, 512)
s = torch.randn(1)
