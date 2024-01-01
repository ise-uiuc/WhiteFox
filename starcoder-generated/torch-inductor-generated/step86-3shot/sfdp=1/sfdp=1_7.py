
class Model(torch.nn.Module):
    def __init__(self, inv_scale_factor, dropout_p, dim):
        super().__init__()
        self.dot_product = torch.nn.MultiheadAttention(dim, 1, dropout=dropout_p, bias=False, add_bias_kv=False)
 
    def forward(self, x1, x2, x3):
        v1 = self.dot_product(x1, x2, x3)
        return v1

# Initializing the model
m = Model(1.1754943508222875e-38, 0.0, 8)

# Inputs to the model
x1 = torch.randn(1, 1, 8)
x2 = torch.randn(1, 1, 8)
x3 = torch.randn(1, 1, 8)
