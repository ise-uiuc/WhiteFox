 definition
def compute_qkv(input):
    heads_count, key_channels, query_channels, value_channels = 4, 8, 8, 8
    w_heads_qkv = torch.nn.Linear(5, 8)

    qkv = w_heads_qkv(input)
    q, k, v = torch.chunk(qkv, 3, dim=-1)
    return q, k, v

m = Model()

# Inputs to the model
x = torch.randn(2, 5)
_,__,___ = compute_qkv(x)
__, ___, ____ = compute_qkv(x)

