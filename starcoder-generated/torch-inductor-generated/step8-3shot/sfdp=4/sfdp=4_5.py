
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, q, k, v, masking):
        scale = math.sqrt(k.size(-1))
        q /= scale
        k /= scale
        attn_mask = (1 - masking[..., None, None].transpose(-2, -1)) * -10000.
        logits = torch.matmul(q, k.transpose(-2, -1))
        logits += attn_mask
        attn_weights = F.softmax(logits, -1)
        return torch.matmul(attn_weights, v), attn_weights

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(5, 3, 90)
k = torch.randn(5, 5, 50)
v = torch.randn(5, 5, 10)
masking = torch.zeros(5, 5)

__output__, __attn_weights__ = m(q, k, v, masking)
