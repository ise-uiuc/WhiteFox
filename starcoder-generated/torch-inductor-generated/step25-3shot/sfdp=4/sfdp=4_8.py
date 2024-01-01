
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
        
    def forward(self, q, k, v, attn_mask):
        attn_weight = F.softmax((q @ k.transpose(-2, -1)), dim=-1)
        attn_weight *= attn_mask
        output = (attn_weight @ v)
        output *= attn_mask
        return q, k, v, attn_mask, attn_weight, output

# Initializing the model using values of dummy tensors
q = torch.randn(1, 3, 64, 64)
k = torch.randn(1, 3, 64, 64)
v = torch.randn(1, 3, 64, 64)
attn_mask = torch.ones(1, 3, 64, 64).bool()
m = Model()

# Inputs to the model
q, k, v, attn_mask = m(q, k, v, attn_mask)

