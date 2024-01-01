
scale_factor = 0.125
dropout_p = 0.1
num_heads = 8
input_depth = 3
projection_depth = 32
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1):
        q = self.q(x1).view(-1, num_heads, input_depth // num_heads, (input_depth // num_heads) * 3)
        k = self.k(x1).view(-1, num_heads, input_depth // num_heads, (input_depth // num_heads) * 3)
        v = self.v(x1).view(-1, num_heads, input_depth // num_heads, (input_depth // num_heads) * 3)
        scale_factor = 1 / math.sqrt(input_depth // num_heads)
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(v).view(-1, num_heads * (input_depth // num_heads), 3 * 3)
        return output

    @torch.jit.export
    def q(self, x1):
        q = torch.nn.functional.max_pool2d(x1, (3,3), stride=(2, 2))
        return q
 
    @torch.jit.export
    def k(self, x1):
        q = torch.nn.functional.avg_pool2d(x1, (3,3), stride=(2, 2))
        return q
 
    @torch.jit.export
    def v(self, x1):
        q = torch.nn.functional.adaptive_avg_pool2d(x1, (3,3))
        return q

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
