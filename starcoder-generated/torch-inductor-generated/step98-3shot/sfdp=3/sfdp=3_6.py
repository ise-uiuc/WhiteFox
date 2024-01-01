
class Model(torch.nn.Module):
    def __init__(self, hidden_size, num_heads, dropout_p=0.0):
        super().__init__()
        self.query = torch.nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.key = torch.nn.Parameter(torch.randn(hidden_size, hidden_size))

    def forward(self, query, key):
        scale_factor = torch.tensor([[key.size(-1)]], device=key.device) ** -0.5
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)

        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(self.v)
        return output

# Initializing the model
hidden_size = 1024
num_heads = 16
dropout_p = 0.1

m = Model(hidden_size, num_heads, dropout_p)

# Inputs to the model
query = torch.randn(1, 32, hidden_size)
key = torch.randn(1, 64, hidden_size)
value = torch.randn(1, 64, hidden_size)
