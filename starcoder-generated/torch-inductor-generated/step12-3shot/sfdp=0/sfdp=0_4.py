
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.key = torch.nn.Parameter(torch.randn(9, 5, 7))
    def forward(self, inp):
        q = inp # x1
        k = inp # x1
        v = inp # x1
        inv_scale = math.sqrt(inp.size(2))
        scaled_dot_product = torch.matmul(q, k.transpose(-2, -1)) / inv_scale
        attention_weights = scaled_dot_product.softmax(dim=-1)
        output = attention_weights.matmul(v)
        return output
# Inputs to the model
x1 = torch.randn(1, 3, 8, 64, 64) # batch, channel, 3-D tensor
