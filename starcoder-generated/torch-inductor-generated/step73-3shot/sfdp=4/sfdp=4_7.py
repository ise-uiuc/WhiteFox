
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
    def forward(self, queries, keys, values, mask):
        QK = queries @ keys.transpose(-2, -1) / math.sqrt(queries.size(-1))
        QK = QK + mask
        attention_scores = torch.softmax(QK, dim=-1)
        context = attention_scores @ values
        return context
# Inputs to the model
Q = torch.randn(1, 64, 56, 56)
K = torch.randn(1, 64, 56, 56)
V = torch.randn(1, 64, 56, 56)
mask = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
