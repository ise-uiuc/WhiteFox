
class Model(torch.nn.Module):
    def __init__(self, d_qkv, d_model, n_heads, dropout_p=0.1):
        super().__init__()
        self.d_qkv = d_qkv
        self.d_model = d_model
        self.dropout_p = dropout_p
        self.n_heads = n_heads
        self.w_qkv = torch.nn.Linear(d_model, d_qkv * n_heads * 3, bias=False)

    def forward(self, x):
        qkv = self.w_qkv(x) # Separate the query, key, and value by using kernel with size 1 to the input tensor
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        s = self.d_qkv ** -0.5
        q *= s
        v *= s
        output = torch.matmul(q, k.transpose(-2, -1))
        scaled_output = output * (self.d_qkv ** -0.5) # Compute the dot product of the query and key tensors
        softmax_output = scaled_output.softmax(dim=-1) # Apply softmax to the scaled dot product
        dropout_output = torch.nn.functional.dropout(softmax_output, p=self.dropout_p) # Apply dropout to the softmax output
        output = dropout_output.matmul(v) # Compute the dot product of the dropout output and the value tensor
        return output

# Initializing the model
d_model = 128
n_heads = 8
m = Model(d_qkv=d_model // n_heads, d_model=d_model, n_heads=n_heads)

# Inputs to the model
x = torch.randn(1, 4, 128)
