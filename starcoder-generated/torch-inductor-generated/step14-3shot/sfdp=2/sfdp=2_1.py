
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2):
        k = self.key(x2)
        q = self.query(x1)
        scale_factor = float(q.size(-1)) ** -0.5
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk * scale_factor
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.5)
        o_t = (dropout_qk * v).sum(dim=-2)
        a = self.att_dense(o_t)
        sa = torch.sigmoid(a)
        c = self.output_dense(sa * x1)
        return c


# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 64, 64)
