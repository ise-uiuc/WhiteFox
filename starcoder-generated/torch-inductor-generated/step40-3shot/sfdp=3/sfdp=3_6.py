
class Model(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
     
    def forward(self, q, k, v, mask):
        qk = torch.matmul(q, k.transpose(-2, -1))
        scale_factor = 1 / math.sqrt(self.dim)
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(v)
        return output


# Initializing the model
m = Model(self.dim)

# Inputs to the model
q = torch.randn(self.bsz, q_len, self.dim)
k = torch.randn(self.bsz, kv_len, self.dim)
v = torch.randn(self.bsz, kv_len, self.dim)
mask = torch.ones(self.bsz, q_len, kv_len)
_ = m(q, k, v, mask)

