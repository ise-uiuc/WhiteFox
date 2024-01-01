
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query = Parameter(torch.randn(2, 3, 5))
        self.key = Parameter(torch.randn(2, 5, 7))
        self.value = Parameter(torch.randn(2, 3, 7))
        self.dropout_p = 0.5
 
    def forward(self, x1):
        qk = self.query @ self.key.transpose(-2, -1) / math.sqrt(self.query.size(-1))
        qk = qk + x1
        attn_weight = torch.softmax(qk, dim=-1)
        attn_weight = torch.dropout(attn_weight, self.dropout_p, True)
        return attn_weight @ self.value

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 1, 2, 3) * -20
