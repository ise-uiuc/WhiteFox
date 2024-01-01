
# Please use the PyTorch library to generate this attention kernel
# The library can be downloaded from https://pytorch.org/get-started/locally/#anaconda-1-5-and-above

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, query, key, value, attn_mask):
        qk = torch.softmax(query * key, 3)
        qk = qk + attn_mask
        attn_weight = torch.dropout(qk, 0.1, True)
        output = attn_weight * value
        return output
# Inputs to the model
query = torch.randn(1, 160, 160, 256)
key = torch.randn(1, 160, 160, 256)
value = torch.randn(1, 160, 160, 256)
attn_mask = torch.randn(1, 1, 160, 160)
