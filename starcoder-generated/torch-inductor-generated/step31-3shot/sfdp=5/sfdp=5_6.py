
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.MultiheadAttention(1024, 8192)
        self.gru = nn.GRU(1024, 1024, 1, batch_first = True)
    def forward(self, x):
        o, w = self.attn(x, x, x, need_weights = True)
        o, _ = self.gru(o)
        return o, w
# Inputs to the model
x = torch.randn(1, 16, 1024)
