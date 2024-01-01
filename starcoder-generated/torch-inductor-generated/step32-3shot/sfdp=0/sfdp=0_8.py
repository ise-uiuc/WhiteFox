
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.k = torch.nn.Parameter(torch.randn(4, 2, 3, 7, 3, 9, 9, 7, 9))
        self.v = torch.nn.Parameter(torch.randn(8, 7, 4, 6, 5, 5, 4, 9, 8))
    def forward(self, x1):
        k = x1
        v = x1
        k = k.reshape(2, 4) # (k.size(0), -1)
        v = v.reshape(8, 56) # (v.size(0), -1)
        q = x1
        q = q.reshape(8, 36) # (q.size(0), -1)
        inv_scale = math.sqrt(k.size(1))
        scaled_dot_product = torch.matmul(q, k.transpose(-2, -1)) / inv_scale
        attention_weights = scaled_dot_product.softmax(dim=-1)
        output = attention_weights.matmul(v)
        return output
# Inputs to the model
x1 = torch.randn(8, 4, 2, 3, 7, 3, 9, 9, 7, 9)
