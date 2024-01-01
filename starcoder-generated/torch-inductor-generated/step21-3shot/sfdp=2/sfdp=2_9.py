
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, x3, x4):
        q = x1
        k = x2
        v = x3
        s = x4
        qk = torch.matmul(q, k.transpose(-2, -1))
        inv_scale_factor = torch.sqrt(q.size(-1)).to(s)
        qk_scaled = qk.div(inv_scale_factor)
        softmax_qk = qk_scaled.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.2)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 2, 1024)
x2 = torch.randn(1, 2, 1024)
x3 = torch.randn(1, 2, 1024)
x4 = torch.randn(1, 2, 1024)
