
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, k2, v2, mask):
        qk = qk @ k2.transpose(-2, -1) / math.sqrt(x.size(-1))
        qk = qk + mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ v2
        return output
# Inputs to the model
Q @ K.transpose(-2, -1) / math.sqrt(Q.size(-1)) -> QK1
Q @ K.transpose(-2, -1) / math.sqrt(Q.size(-1)) + mask -> QK2
QK2 -> QK3
QK3 -> QKD3
QK3 -> softmax
QK3 @ K.transpose(-2, -1) / math.sqrt(Q.size(-1)) -> QKD3

QK = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1)) -> QKD
QK -> QKD1
QK + attn_mask -> QKD2
QK = QK + attn_mask -> QKD21
QK @ key.transpose(-2, -1) / math.sqrt(query.size(-1)) -> QKD22
QK @ key.transpose(-2, -1) / math.sqrt(query.size(-1)) + attn_mask -> QKD2
QK @ key.transpose(-2, -1) / math.sqrt(query.size(-1)) -> QKD2
QK @ key.transpose(-2, -1) / math.sqrt(query.size(-1)) + attn_mask -> QKD2
QK @ key.transpose(-2, -1) / math.sqrt(query.size(-1)) + attn_mask -> QKD3
QK @ key.transpose(-2, -1) / math.sqrt(query.size(-1)) + attn_mask -> QKD4
QK @ key.transpose(-2, -1) / math.sqrt(query.size(-1)) + attn_mask -> QKD5
QK @ key.transpose(-2, -1) / math.sqrt(query.size(-1)) + attn_mask -> QKD6
QK @ key.transpose(-2, -1) / math.sqrt(query.size(-1)) + attn_mask -> QKD7
QK @ key.transpose(-2, -1) / math.sqrt(query.size(-1)) + attn_mask -> QKD8
QK @ key.transpose(-2, -1) / math.sqrt(query.size(-1)) + attn_mask -> QKD9
Q = torch.randn(1, 64, 56, 56)
K = torch.randn(1, 64, 56, 56)
V = torch.randn(1, 64, 56, 56)
mask = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
