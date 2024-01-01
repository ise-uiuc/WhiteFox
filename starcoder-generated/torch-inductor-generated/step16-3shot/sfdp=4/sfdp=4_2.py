
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query = torch.nn.Linear(4, 4)
        self.key = torch.nn.Linear(4, 4)
        self.value = torch.nn.Linear(4, 4)
 
    def forward(self, query, key, value):
        qk = self.query(query) @ self.key(key).transpose(-2, -1) / math.sqrt(self.query.weight.shape[-1])
        qk = qk + attention_mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ self.value(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(3, 5, 4)
key = torch.randn(3, 1, 4)
value = torch.randn(3, 1, 4)
attention_mask = torch.zeros(3, 5, 1, dtype=torch.long)
