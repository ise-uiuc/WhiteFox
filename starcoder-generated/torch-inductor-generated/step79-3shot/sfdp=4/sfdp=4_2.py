
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, QUERY, KEY, VALUE):
        qk = QUERY @ KEY.transpose(-2, -1) / math.sqrt(QUERY.size(-1))
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ VALUE
        return output
# Inputs to the model
QUERY = torch.randn(1, 64, 56, 56)
KEY = torch.randn(1, 64, 56, 56)
VALUE = torch.randn(1, 64, 56, 56)
