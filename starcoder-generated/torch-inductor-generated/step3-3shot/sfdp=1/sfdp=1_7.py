
class Model(torch.nn.Module):
    def __init__(self, batch_dim, num_heads, k, t):
        super().__init__()
        self.batch_dim = batch_dim
        self.num_heads = num_heads
        self.head_dim = k
        self.qk_dim = k * num_heads
        self.dropout_p = t
 
    def forward(self, query, key, value):
        # Flatten the tensor dimensions
        h, r = query.shape[2], key.shape[3]
        query = query.reshape([self.batch_dim, self.qk_dim, h*r])
        key = key.reshape([self.batch_dim, self.qk_dim, h*r])
        value = value.reshape([self.batch_dim, self.qk_dim, h*r])

        # Compute the dot product of the query and key matrices
        scaled_qk = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
 
        # Compute the dot product of the dropout output and value matrices
        output = torch.matmul(dropout_qk, value)
        return output.reshape([self.batch_dim, self.num_heads, k, h, r]).swapaxes(3, 4)

# Initialize the model
__batch_shape__ = [2, 2, 8, 64, 64]  # Batch Shape
num_heads = 2
k = 32
dropout_p = 0.25
m = Model(*__batch_shape__, num_heads=num_heads, k=k, t=dropout_p)

# Inputs to the model
key = torch.randn(*__batch_shape__, num_heads=num_heads, k=k)
value = key
query = key
