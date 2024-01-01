
class Model(torch.nn.Module):
    def __init__(self, d_hid):
        super().__init__()
        self.attn = torch.nn.Linear(d_hid, 128)
 
    def forward(self, q, k):
        q = self.attn(q)
        k = self.attn(k)
        qk = torch.bmm(q., k.transpose(1, 2))
        inv_scale_factor = (q.shape[-1] * k.shape[-2]).div(16 ** 2)
        qk = qk.div(inv_scale_factor)
        return qk.softmax(dim=-1)

# Initializing the model
m = Model(16)

# Inputs to the model
q = torch.randn(4, 2, 16)
k = torch.randn(4, 4, 16)
