
class Model(torch.nn.Module):
    def __init__(self, d_model, heads, dropout_p=0.0):
        super().__init__()
        self.m = d_model // heads
        self.scale_factor = d_model ** -0.5
        self.dropout = torch.nn.Dropout(dropout_p)
        self.dot_product = ScaledDotProductAttention(self.m, self.scale_factor, self.dropout)
 
    def forward(self, k, v, q):
        output, weights = self.dot_product(k, v, q)
        return output

# Initializing the model
m = Model(d_model=4, heads=2)

# Inputs to the model
k = torch.randn(5, 1, 4)
v = torch.randn(5, 1, 4)
q = torch.randn(1, 1, 4)
__output__, __weights__ = m(k, v, q)

