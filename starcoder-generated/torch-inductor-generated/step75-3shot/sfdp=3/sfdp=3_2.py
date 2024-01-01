
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v):
        query = v1 = torch.nn.functional.linear(q, self.query_weight, self.query_bias)
        key = v2 = torch.nn.functional.linear(k, self.key_weight, self.key_bias)
        value = torch.nn.functional.linear(v, self.value_weight, self.value_bias)
        scale_factor = self.scale_factor
        dropout_p = self.dropout_p
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model(32, 50, 10, 0.5, 0.8)

# Inputs to the model
q = torch.randn(1, 32, 50)
k = torch.randn(1, 32, 100)
v = torch.randn(1, 32, 100)
