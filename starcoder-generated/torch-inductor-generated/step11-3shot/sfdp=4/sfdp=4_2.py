
class Model(torch.nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.linear_query = torch.nn.Linear(3, hidden)
        self.linear_key = torch.nn.Linear(3, hidden)
        self.linear_value = torch.nn.Linear(3, hidden)
        self.attn_drop = nn.Dropout(0.0)
        self.proj = torch.nn.Linear(hidden, 3)
 
    def forward(self, x1, x2):
        v1 = self.linear_query(x1)
        v2 = self.linear_key(x2)
        v3 = self.linear_value(x2)
        qk = q @ k.transpose(-2, -1) / math.sqrt(query.size(-1))
        qk += attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        attn = attn_weight @ v
        attn = self.attn_drop(attn)
        output = self.proj(attn)
     
        return output, v, attn_weight

# Initializing the model
m = Model(3)

# Inputs to the model
x1 = torch.randn(3, 3)
x2 = torch.randn(3, 3)
attn_mask = torch.zeros(3, 3)
__output__, __output__v, __output__attn_weight = m(x1, x2)

