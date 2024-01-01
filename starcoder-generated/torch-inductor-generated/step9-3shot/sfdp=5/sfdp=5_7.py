
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query = torch.nn.Linear(1, 1)
        self.key = torch.nn.Linear(2, 2)
        self.value = torch.nn.Linear(3, 3)
 
    def forward(self, x4, x5):
        v4 = self.query(x4)
        v5 = self.key(x5)
        qk = v4 @ torch.transpose(v5, -2, -1) / math.sqrt(v4.size(-1))
        qk = qk + self.attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        attn_weight = torch.dropout(attn_weight, 0.8, True)
        output = attn_weight @ self.value
        return output

# Initializing the model
m = Model()

# Inputs to the model
x4 = torch.randn(1, 1)
x5 = torch.randn(2, 2)
