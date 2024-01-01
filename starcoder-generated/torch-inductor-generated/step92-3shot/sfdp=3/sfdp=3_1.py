
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = Attention(8)

    def forward(self, q, k, v, scale_factor=1, dropout_p=0.):
        self.attn_weights = self.attn(q, k, v, scale_factor, dropout_p)

        return self.attn_weights

# Initializing the model
m = Model()

# Inputs to the model
queries = torch.randn(12, 30, 8)
keys = torch.randn(12, 20, 8)
values = torch.randn(12, 20, 64)
input_scale_factor = m.attn.get_scale_factor()
input_dropout_p = m.attn.get_dropout_p()
