
def scaled_dot_product_attention(query, key, value, scale_factor):
   qk = torch.matmul(query, key.transpose(-2, -1))
   scaled_qk = qk.div(scale_factor)
   softmax_qk = scaled_qk.softmax(dim=-1)
   dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.2)
   output = dropout_qk.matmul(self.value)
   return output

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale_factor = torch.nn.Parameter(torch.tensor(1.0))
 
    def forward(self, query, key, value):
        v1 = scaled_dot_product_attention(query, key, value, self.scale_factor)
        return v1

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8, 128)
x2 = torch.randn(1, 16, 128)
x3 = torch.randn(1, 16, 128)
