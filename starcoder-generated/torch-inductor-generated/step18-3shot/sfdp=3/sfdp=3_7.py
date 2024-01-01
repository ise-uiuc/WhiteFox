
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.d_k = 64
        self.h = 64
        self.w = 64
 
    def forward(self, x1, x2):
        q = x1
        k = x2
        qk = torch.matmul(q, k.transpose(-2, -1))
        scale_factor = q.size(-1) ** 0.5
        softmax_qk = qk.mul_(scale_factor).softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.0003)
        v = x2
        vq = dropout_qk.matmul(v)
        return vq

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 64, 64)
