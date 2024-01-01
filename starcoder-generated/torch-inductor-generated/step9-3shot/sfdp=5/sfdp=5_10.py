
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q = torch.nn.Linear(4, 1)
        self.k = torch.nn.Linear(4, 1)
        self.v = torch.nn.Linear(4, 1)
 
    def forward(self, x1):
        q, k, v = self.q(x1), self.k(x1), self.v(x1)
        qk = torch.bmm(q, k.transpose(-1, -2))
        attn_mask = torch.zeros(qk.shape, dtype=qk.dtype, device=qk.device)
        attn_mask = torch.where(qk!= float("-inf"), float("-inf"), attn_mask)
        attn_weight = torch.softmax(qk + attn_mask, dim=-1)
        attn_weight = torch.dropout(attn_weight, 0.0)
        output = torch.bmm(attn_weight, v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 2, 4)
