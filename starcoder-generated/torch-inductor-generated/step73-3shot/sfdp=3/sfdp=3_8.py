
class Model(torch.nn.Module):
    def __init__(
        self,
        num_heads,
        dim,
        dropout_p,
        num_batches
    ):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.dropout_p = dropout_p
        self.num_batches = num_batches
 
    def forward(self, query, key, value, scale_factor):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(
                softmax_qk, 
                p=self.dropout_p, 
                training=self.training
        )
        v = dropout_qk.matmul(value)
        return v

# Initializing the model
num_heads = 4
dim = 8
dropout_p = 0.5
num_batches = 1
m = Model(num_heads, dim, dropout_p, num_batches)

# Inputs to the model
input_length = 64
scale_factor = 1 / math.sqrt(dim)
query = torch.randn(input_length, num_heads, dim)
key = torch.randn(input_length, num_heads, dim)
value = torch.randn(input_length, num_heads, dim)
