
class Model(torch.nn.Module):
    def __init__(self, in_features, num_heads, num_keys, key_size):
        super().__init__()
        self.query = torch.nn.Linear(in_features, key_size * num_heads)
        self.key = torch.nn.Linear(num_keys, key_size * num_heads)
        self.scale_factor = torch.sqrt(torch.Tensor([key_size]))
        self.dropout = torch.nn.Dropout(0.1)
 
    def forward(self, x1, x2):
        q = self.query(x1)
        q = q.reshape(q.shape[0], -1, q.shape[-1])
        k = self.key(x2)
        k = k.reshape(k.shape[0], -1, k.shape[-1])
        qk = torch.matmul(q, k.transpose(-1, -2))
        scaled_qk = qk * self.scale_factor
        softmax_qk = softmax(scaled_qk, dim=-1)
        dropout_qk = self.dropout(softmax_qk)
        output = torch.matmul(dropout_qk, v)
        output = output.reshape(-1, output.shape[-2] * output.shape[-1])
        return output

# Initializing the model
m = Model(10, 4, 120, 20)

# Inputs to the model
x1 = torch.randn(4, 10)
x2 = torch.randn(15, 10)
