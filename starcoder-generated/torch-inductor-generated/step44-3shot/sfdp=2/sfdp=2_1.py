
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv_proj = torch.nn.Linear(6, 25, bias=False)
 
    def forward(self, x1):
        qkv = F.gelu(self.qkv_proj(x1))
        qkv = qkv.split(25, dim=-1)
        q = qkv[0]
        k = qkv[1]
        v = qkv[2]
        q = q.unsqueeze(dim=-3)
        q = q.expand_as(k)
        scaled_qk = torch.matmul(q, k.transpose_(-2, -1))
        inv_scale_factor = 1. / math.sqrt(math.sqrt(6))
        scaled_qk = scaled_qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.5)
        output = torch.matmul(dropout_qk, v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 6, 4)
