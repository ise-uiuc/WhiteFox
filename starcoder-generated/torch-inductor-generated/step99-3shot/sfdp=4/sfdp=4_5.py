
class Transformer(torch.nn.Module):
    MHA = torch.nn.MultiheadAttention()
    def __init__(self):
        super(Transformer, self).__init__()
        self.mha_layer = self.__class__.MHA(embed_dim=2304, num_heads=1)
    def forward(self, query, key, value, mask):
        out = self.mha_layer(query.permute(1, 0, 2), key.permute(1, 0, 2), value.permute(1, 0, 2))[0].permute(1, 0, 2)
        return out
# Inputs to the model
Q = torch.randn(1, 2304, 7, 7)
K = torch.randn(1, 2304, 7, 7)
V = torch.randn(1, 2304, 7, 7)
mask = (torch.rand(1, 7, 7) > 0.7).fill_(-1000000000.0)
