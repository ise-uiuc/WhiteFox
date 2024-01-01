
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = 64
        self.seq_len = 128
        self.dim = 1
    def forward(self, query, key, value, attn_mask, bias=None):
        qk = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        attn_weight = torch.dropout(attn_weight, 0.2, True)
        output = attn_weight @ value
        output = output.__neg__()
        if bias is not None:
            assert bias.dim() == 1
            output += query + torch.softmax(bias.unsqueeze(1).unsqueeze(0), dim=-1)
        return output
# Inputs to the model
query = torch.nn.parameter.Parameter(torch.randn_like(torch.rand(1, 64, 128, 1))).to(torch.int64)
key = torch.nn.parameter.Parameter(torch.randn_like(torch.rand(1, 64, 128, 1))).to(torch.int64)
value = torch.nn.parameter.Parameter(torch.randn_like(torch.rand(1, 64, 128, 1))).to(torch.int64)
attn_mask = torch.randn(1, 1, 128, 128)
bias = torch.nn.parameter.Parameter(torch.randn_like(torch.rand(64)))
