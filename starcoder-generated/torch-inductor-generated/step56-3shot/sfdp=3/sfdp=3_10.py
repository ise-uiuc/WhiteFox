
class Model(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(Model).__init__()
        self.weight = torch.nn.Parameter(torch.randn(input_size, output_size))
 
    def forward(self, q, k, v, dropout_p):
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.mul(0.1)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
input_size = 20
output_size = 30
dropoutp = 0.1
m = Model(input_size, output_size)

# Inputs to the model
q = torch.randn(1, 2, input_size)
k = torch.randn(1, 10, input_size)
v = torch.randn(1, 10, output_size)
__output(m(q, k, v)

# Note that in this example, the query, key, and value tensors have different input and output size for demonstration purpose. However, in most cases, they would have the same input and output size so there's no need to pay attention to this. In the following case, the input tensor of the key would be `query.transpose(-2, -1)`.

# Model
class Model(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(Model).__init__()
        self.weight = torch.nn.Parameter(torch.randn(input_size, output_size))
 
    def forward(self, q, k, v, dropout_p):
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.mul(0.1)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
input_size = 20
output_size = 30
dropoutp = 0.1
m = Model(input_size, output_size)

# Inputs to the model
q = torch.randn(1, 2, input_size)
k = torch.randn(1, 2, input_size)
v = torch.randn(1, output_size)
output = __output(m(q, k, v)

