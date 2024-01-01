
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, self_attn_layer_norm, qk, mask):
        x = self_attn_layer_norm(x)
        qk = qk + mask
        attn_weight = torch.softmax(qk, dim=-1)
        q = attn_weight @ v
        return x + q
