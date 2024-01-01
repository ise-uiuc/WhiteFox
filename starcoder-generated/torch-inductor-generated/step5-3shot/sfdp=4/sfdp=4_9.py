
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(query, key, value, attn_mask):
        qk = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ value
        return output
 
# Initializing the model
m = Model()

# Inputs to the model
__query__ = torch.rand(10, 3, 20)
__key__ = torch.rand(10, 4, 20)
__value__ = torch.randn(10, 4, 20)
__attn_mask__ = torch.tensor([[0, 10, 0],[0, 0, 0],[0, 0, 0]], dtype=torch.float32)
