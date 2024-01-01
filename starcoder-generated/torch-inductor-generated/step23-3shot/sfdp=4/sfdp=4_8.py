
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(3, 8)
 
    def forward(self, query, key, value, attn_mask):
        qk = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ value
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 3, 128)
key = torch.randn(1, 3, 512)
value = torch.randn(1, 3, 512)
attn_mask = torch.randn(1, 1, 1, 512)
