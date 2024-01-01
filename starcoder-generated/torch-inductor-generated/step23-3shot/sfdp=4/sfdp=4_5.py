
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)
 
    def forward(self, qk, attn_mask):
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ self.linear.weight
        return output

# Initializing the model
m = Model()

# Inputs for the model
qk = torch.randn(1, 1, 1)
attn_mask = torch.zeros(1, 1, 1)
_value = m(qk, attn_mask)

