
class Model(torch.nn.Module):
    scaled_keys = None
 
    def __init__(self, p, dropout_p, q, key, value, scale_factor):
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout_p)
        self.softmax = torch.nn.Softmax(q, key)
        self.scaled_dot_product = self.softmax.mul(scale_factor)
 
    def forward(self, __x1__, __x2__):
        qk = torch.matmul(__x1__, __x2__.transpose(-2, -1))
        v = qk.matmul(__x2__)
        v = self.scaled_dot_product.matmul(v)
        v = self.dropout(v)
        return v

# Inputs to the model
p = 0
dropout_p = 0.5
q = 1
key = torch.randn(1, 3, 64, 64)
value = torch.randn(1, 3, 64, 64)
scale_factor = torch.randn(1)
