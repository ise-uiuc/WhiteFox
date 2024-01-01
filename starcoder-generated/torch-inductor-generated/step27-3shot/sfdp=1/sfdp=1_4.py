
class Model(torch.nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads
 
    def forward(self, query, key, value, scale_factor, dropout_p):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(scale_factor)
        softmax_qk = torch.nn.functional.softmax(scaled_qk, dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        output = output.contiguous().view(query.size(0), -1, self.num_heads * value.size(-1)).transpose(1, 2)
        return output

# Initializing the model
m = Model(num_heads=4)

# Inputs to the model
N, T = 1, 1
query = torch.randn(N, T, 4, 4)
key = torch.randn(N, T, 4, 4)
value = torch.randn(N, T, 4, 4)
scale_factor = 1.0 / np.sqrt(4)
dropout_p = 0.5
