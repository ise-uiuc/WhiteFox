
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = torch.nn.Softmax(dim=-1)
 
    def forward(self, *inputs):
        q, k, v, mask, dropout_p = inputs
        qk = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
        qk = qk + mask
        attn_weight = self.softmax(qk)
        attn_weight = torch.dropout(attn_weight, dropout_p, True)
        output = attn_weight @ v
        return output, attn_weight

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(1, 1, 16)
k = torch.randn(1, 1, 64)
v = torch.randn(1, 64, 64)
mask = torch.zeros(1, 1, 16, 64).bool()
dropout_p = 0.1
__output__, __state__ = m(q, k, v, mask, dropout_p)

