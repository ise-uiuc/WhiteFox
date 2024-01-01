
class Model(torch.nn.Module):
    def __init__(self, dropout_p, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor
        self.attn = torch.nn.MultiheadAttention(3, 3, dropout=dropout_p)
 
    def forward(self, x1, x2, x3):
        x4, v1, v2 = x1, x2, x3
        v1a = x2*self.scale_factor
        v2a = x3*self.scale_factor
        v3, v1b = self.attn(v1a, v2a, v2a)
        v4 = v3 * v1
        return x4, v1b, v4

# Initializing the model with dropout probability 0.5 and scale factor 1/sqrt(32)
scale_factor = 1/np.sqrt(32)
dropout_p = 0.5
input_tensor = torch.randn(1, 32, 3, 3)
value_tensor = torch.randn(1, 32, 3, 3)
key_tensor = torch.randn(1, 32, 3, 3)
__out1__, __out2__, __out3__ = m(input_tensor, value_tensor, key_tensor)

