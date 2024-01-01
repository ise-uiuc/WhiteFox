
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = Attention(3)
 
    def forward(self, query, key, value):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        output = softmax_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 3, 64, 64)
x2 = torch.randn(2, 3, 64, 64)
x3 = torch.randn(2, 3, 64, 64)
