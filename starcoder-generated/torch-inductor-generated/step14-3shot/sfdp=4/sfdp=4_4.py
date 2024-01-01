
class Model(torch.nn.Module):
    def __init__(self, attention_head_size=8):
        super().__init__()
        self.attention_head_size = attention_head_size
        self.q = torch.nn.Linear(768, attention_head_size)
        self.k = torch.nn.Linear(768, attention_head_size)
        self.v = torch.nn.Linear(768, attention_head_size)
        self.o = torch.nn.Linear(attention_head_size, 768)

    def forward(self, query, key, value, attn_mask):
        q = self.q(query)
        k = self.k(key)
        v = self.v(value)
        qk = q @ k.transpose(-2, -1) / math.sqrt(k.size(-1))
        qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ v
        output = self.o(output)
        return output
        
# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 64, 768)
key = torch.randn(1, 64, 768)
value = torch.randn(1, 64, 768)
attn_mask = torch.ones(1, 64, 64)
