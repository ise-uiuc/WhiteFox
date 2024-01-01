
class Model(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = torch.nn.MultiheadAttention(dim, 1, dropout=p)
 
    def forward(self, x1, x2, x3, mask):
        query = x1
        key = x2
        value = x3
        qk = self.attn.forward_query(query, key)
        dk = self.attn.forward_key(query, key)
        dv = self.attn.forward_value(key)
        dk = dk * ((float(mask.size(1)) / mask.sum(-1, keepdim=True)) ** 0.5)
        qk = qk.div((float(mask.size(0)) / mask.sum(-2, keepdim=True)) ** 0.5)
        scaled_qk = qk.softmax(-1)
        dropout_qk = self.attn.dropout_module(scaled_qk)
        output = dropout_qk.matmul(dv.transpose(-2,-1))
        return output

# Initializing the model
m = Model(dim)

# Inputs to the model
x1 = torch.randn(1, 64, dim)
x2 = torch.randn(1, 100, dim)
x3 = torch.randn(1, 100, dim)
mask = torch.ones(size=(1, 100), dtype=bool)
