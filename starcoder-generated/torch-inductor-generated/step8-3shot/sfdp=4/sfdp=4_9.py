
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):        
        qk = x1 @ x2.transpose(-2, -1) / math.sqrt(x1.size(-1))        
        qk = qk + attn_mask        
        attn_weight = torch.softmax(qk, dim=-1)        
        output = attn_weight @ x3        
        return output

# Initializing the model
q = torch.randn(1, 1, 10)
k = torch.randn(1, 3, 10).transpose(-2, -1)
v = torch.randn(1, 3, 10)
attn_mask = torch.triu(torch.ones(3, 3))
m = Model()

# Inputs to the model
