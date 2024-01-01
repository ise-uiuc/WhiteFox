
# TODO: modify this model to remove the "None"-type operands
# TODO: what is the input shape?
# TODO: should the last dim of the value and last dim of the key be the same?
# TODO: the shape of "attn_weights" is different, can you think of reason for the different shapes?
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.key = torch.nn.Parameter(torch.randn(4, 3, 2, 3))
    def forward(self, x1):
        q = x1
        k = self.key
        v = self.key
        inv_scale = math.sqrt(k.size(-1))
        scaled_dot_product = torch.matmul(q, k.transpose(-2, -1)) / inv_scale
        attn_weights = scaled_dot_product.softmax(dim=-2)
        attn_weights1 = attn_weights
        attn_weights2 = attn_weights
        attn_weights3 = attn_weights
        output = attn_weights1.matmul(v) + attn_weights2.matmul(v) + attn_weights3.matmul(v)
        return output
# Inputs to the model
x1 = torch.randn(1, 4, 64, 64)
