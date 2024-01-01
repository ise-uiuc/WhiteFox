
class Model(torch.nn.Module):
    def __init__(self, dropout_p=0.42):
        super().__init__()
        self.dropout_p = dropout_p

    def _generate_bias(self, query):
        batch_size = query.shape[0]
        num_heads = query.shape[1]
        n = int(query.shape[2])
        return torch.randn(num_heads, batch_size, n, n)

    def forward(self, query, key, value, scale_factor):
        bias = self._generate_bias(query)
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        d_q = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = d_q.matmul(value)
        return output

# Initializing the model
query = torch.randn(16, 8, 512, 64)
key = torch.randn(16, 8, 64, 512)
value = torch.randn(16, 8, 512, 64)
scale_factor = torch.randn(16, 8, 1, 1)

m = Model()
