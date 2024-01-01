
class Model(torch.nn.Module):
    def __init__(self, dim, dropout_p=0.1):
        super().__init__()
        self.dim = dim
        self.dropout_p = dropout_p

    def forward(self, query, key, value):
        inv_scale_factor = torch.rsqrt(torch.tensor(self.dim).float())
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk * inv_scale_factor
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model(512)

# Inputs to the model
query = torch.randn(1, 10, 512)
key = torch.randn(1, 100, 512)
value = torch.randn(1, 100, 512)
