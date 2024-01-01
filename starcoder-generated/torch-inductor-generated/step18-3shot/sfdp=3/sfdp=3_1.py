
def scaled_dot_product_attention(query, key, value, scale_factor=1, dropout_p=0.0):
    qk = torch.matmul(query, key.transpose(-2, -1))
    scaled_qk = qk.mul(scale_factor)
    softmax_qk = scaled_qk.softmax(dim=-1)
    dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
    output = dropout_qk.matmul(value)
    return output

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query = torch.nn.Parameter(torch.rand(16, 32, 9, 9))
        self.key = torch.nn.Parameter(torch.rand(16, 32, 11, 11))
        self.value = torch.nn.Parameter(torch.rand(16, 32, 11, 11))
        
    def forward(self, x1):
        return scaled_dot_product_attention(self.query, self.key, self.value)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(8, 16, 64, 64)
