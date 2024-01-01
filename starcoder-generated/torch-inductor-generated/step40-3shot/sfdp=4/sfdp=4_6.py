
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, q4, q5, k3, v4, mask):
        qk = q3 @ k2.transpose(-2, -1) / math.sqrt(q3.size(-1))
        qk = qk + mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ v2
        return output
