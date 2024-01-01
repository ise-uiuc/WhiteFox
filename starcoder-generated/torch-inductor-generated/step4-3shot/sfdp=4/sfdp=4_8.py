
class Model(torch.nn.Module):
    def __init__(self, emb_size, block_size):
        super().__init__()
        
    def forward(self, q, k, v, mask):
        attn_logits = torch.matmul(q, k.transpose(-2, -1))
        attn_logits = attn_logits / math.sqrt(emb_size)
        attn_logits += mask
        normalized_weights = torch.softmax(attn_logits, dim=-1)
        result = torch.matmul(normalized_weights, v)
        return result

# Initializing the model
m = Model(emb_size = 12, block_size = 10)

# Inputs to the model
q = torch.randn(1, 10, 12)
k = torch.randn(1, 20, 12)
v = torch.randn(1, 20, 12)
mask = torch.randn(1, 10, 20)
