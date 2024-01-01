
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = torch.nn.Softmax(dim=-1)
 
    def forward(self, qkv):
        q, kv = torch.split(qkv, [8, 16])
        qk = torch.matmul(q, kv.transpose(-2, -1))
        inv_scale_factor = torch.rsqrt((q * q).sum(dim=-1, keepdim=True) + 1e-12)
        scaled_qk = qk * inv_scale_factor
        softmax_qk = self.softmax(scaled_qk)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.1)
        return dropout_qk

# Initializing the model
m = Model()

# Inputs to the model
qkv = torch.randn(1, 32, 8)
