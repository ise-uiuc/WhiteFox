
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = torch.nn.Linear(37, 38)
 
    def forward(self, x1):
        v0 = x1
        v2 = self.qkv(v0)
        b, t, s = v2.size()
        s = s // 3
        v3 = v2.view(b, t, 3, s)
        q, k, v = v3.unbind(dim=2)
        qk = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
        qk = qk + 1.0 - torch.eye(t, t).to(v2.device)
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ v
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(7, 1, 37)
