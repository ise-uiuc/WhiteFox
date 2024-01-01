
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.m1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.m2 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
 
    def forward(self, x1):
        v1 = self.m1(x1)
        v2 = self.m2(x1)
        q = (v1 + v2) * 0.1
        k = (v1 * v2) * 0.5
        v = (v1 + v2) * 0.2041241452316284
        qk = torch.matmul(q, k)
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
