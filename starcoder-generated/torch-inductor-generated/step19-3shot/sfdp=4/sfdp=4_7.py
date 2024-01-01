
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, attn_mask):
        qk = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
        qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ v
        return output

attn_mask = torch.ones([1, 3, 3]).tril(-1)
m = Model()
q = torch.randn([1, 3, 5, 6])
k = torch.randn([1, 5, 4])
v = torch.randn([1, 5, 6])
output = m(q, k, v, attn_mask)

