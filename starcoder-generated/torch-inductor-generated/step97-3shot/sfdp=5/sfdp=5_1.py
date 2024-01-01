
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = 64
        self.seq_len = 32
        self.dim = 472 // self.heads
    def forward(self, query, key, value, attn_mask):
        qk = query @ key.transpose(-2, -1) #+ query.mean(dim=-2, keepdims=True) @ key.mean(dim=-2, keepdims=True).transpose(-2, -1)
        qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        attn_weight = torch.dropout(attn_weight, 0.0, True)
        output = attn_weight @ value + value
        output1 = attn_weight @ value + value
        output2 = attn_weight @ value 
        output3 = output1 + output2
        return output3
# Inputs to the model
query = torch.randn(1, 64, 32, 472)
key = torch.randn(1, 64, 32, 472)
value = torch.randn(1, 64, 32, 472)
attn_mask = torch.randn(1, 1, 32, 32)
