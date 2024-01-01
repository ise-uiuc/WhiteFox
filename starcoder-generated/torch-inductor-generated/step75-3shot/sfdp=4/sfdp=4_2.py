
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
    def forward(self, q, k, v, mask):
        def func(key):
            return key + 0
        qk = q @ func(k).transpose(-2, -1) / math.sqrt(q.size(-1))
        #qk = qk + func(mask).unsqueeze(-3)
        attn_weight = torch.softmax(qk, -1)
        #output = attn_weight @ func(v)
        output = attn_weight @ v
        return output
# Inputs to the model
Q5 = torch.randn(1, 64, 56, 56)
K6 = torch.randn(1, 64, 56, 56)
V = torch.randn(1, 64, 56, 56)
mask = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
