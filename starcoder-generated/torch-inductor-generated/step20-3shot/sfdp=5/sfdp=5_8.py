
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.max_input_length = 256
        self.dropout = 0.1
        self.heads = 32
        self.dim = 64
    def forward(self, query, key, value, attn_mask):
        query = query.transpose(0, 1)
        key   = key.transpose(0, 1)
        value = value.transpose(0, 1)
        qk = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        qk = qk.transpose(0, 1).contiguous().view(-1, self.heads, (self.max_input_length if len(qk.shape)<6 else qk.size(4)), self.max_input_length) + attn_mask
        attn_weight = torch.softmax(qk.view(-1, qk.size(-3), qk.size(-1)), dim=-1)
        attn_weight = pytorch_utils.dropout(attn_weight, 0.1, True)
        output = attn_weight @ value.transpose(0, 1)
        return output.transpose(0, 1).contiguous().view(-1, output.size(-3), output.size(-2), output.size(-1))
# Inputs to the model
query = torch.randn(256, 32, 64)
key = torch.randn(256, 32, 64)
value = torch.randn(256, 32, 64)
attn_mask = torch.randn(256, 1, 256, 256)
