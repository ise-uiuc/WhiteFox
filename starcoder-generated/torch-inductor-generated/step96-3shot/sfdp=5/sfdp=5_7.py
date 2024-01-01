
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(512, 64)
        self.fc = torch.nn.Linear(1280, 256)
        self.seq_len = 1
    def forward(self, query, key, value, attn_mask):
        query = self.fc1(query)
        key = key.transpose(0, 1).contiguous()
        key = key.view(key.shape[0], -1)
        key = self.fc(key)
        key = key.contiguous().view(key.shape[0], key.shape[1], -1)
        key = key.transpose(0, 1).contiguous()
        qk = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        attn_weight = torch.dropout(attn_weight, dropout, True)
        output = attn_weight @ value
        return output
# Inputs to the model
query = torch.randn(16, 1, 512)
key = torch.randn(1, 16, 512)
value = torch.randn(1, 16, 512)
attn_mask = torch.randn(1, 16, 16)
