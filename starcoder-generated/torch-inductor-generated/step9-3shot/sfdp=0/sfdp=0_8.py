
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query_kernel = torch.nn.Parameter(torch.randn(16, 8, 3, 3), requires_grad=True)
        self.key_kernel = torch.nn.Parameter(torch.randn(16, 3, 3, 3), requires_grad=True)
        self.value_kernel = torch.nn.Parameter(torch.randn(16, 3, 1), requires_grad=True)
        self.inv_sqrt_depth = 1 / math.sqrt(3 * 3)
 
    def scaled_dot_product_attention(self, query, key, value, inv_scale):
        dot = torch.matmul(query, key.transpose(-2, -1))
        attn = dot * inv_scale
        attn = torch.softmax(attn, -1)
        output = attn.matmul(value)
        return output
 
    def forward(self, x1):
        query = torch.nn.functional.conv2d(x1, self.query_kernel, groups=16)
        key = torch.nn.functional.conv2d(x1, self.key_kernel, groups=16)
        value = torch.nn.functional.conv2d(x1, self.value_kernel, groups=16)
        output = self.scaled_dot_product_attention(query, key, value, self.inv_sqrt_depth)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(16, 3, 64, 64)
