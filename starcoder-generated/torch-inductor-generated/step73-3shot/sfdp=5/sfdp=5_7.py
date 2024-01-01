
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query = torch.nn.Linear(768, 3072)
        self.key = torch.nn.Linear(768, 3072)
        self.value = torch.nn.Linear(768, 3072)
        self.attention_mask = torch.nn.Linear(768, 1)
        self.output = torch.nn.Linear(3072, 768)
    def forward(self, query, key, value, attn_mask):
        qk = self.query(query) @ self.key(key).transpose(-2, -1)
        qk = qk / math.sqrt(3072)
        qk = qk + self.attention_mask(attn_mask)
        attn_weights = torch.softmax(qk, dim=-1)
        attn_weights = torch.dropout(attn_weights, 0.1, True)
        output = self.output(attn_weights @ value)
        return output
# Inputs to the model
query = torch.randn(1, 1, 768)
key = torch.randn(1, 1, 768)
value = torch.randn(1, 1, 768)
attn_mask = torch.randn(1, 1, 768)
