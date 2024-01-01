
class Model(torch.nn.Module):
    def __init__(self, query_dim, key_dim, value_dim):
        super().__init__()
        self.scale_factor = key_dim ** 0.5
        self.dropout_p = 0.1
 
    def forward(self, query, key, value):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.mul(self.scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(
            softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(value)
        return output
 
# Initializing the model
m = Model(32, 32, 64)

# Inputs to the model
# Inputs are float tensors with 8 heads in the batch dimension.
query = torch.randn(8, 5, 32)
key   = torch.randn(8, 6, 32)
value = torch.randn(8, 6, 64)
