
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = 1
        self.seq_len = 5
        self.dim = 128
        self.hidden_size=128
        self.input_size=128
    def forward(self, query, key, value, attn_mask):
        qk = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        attn_weight = torch.dropout(attn_weight, 0.2, True)
        output = attn_weight @ value
        q = q + self.fc1(torch.dropout(torch.relu(self.fc0(query))))
        output = output + q
        return output
# Inputs to the model
query = torch.randn(1, 1, 5, 128)
key = torch.randn(1, 1, 5, 128)
value = torch.randn(1, 1, 5, 128)
attn_mask = torch.randn(1, 1, 5, 5)
