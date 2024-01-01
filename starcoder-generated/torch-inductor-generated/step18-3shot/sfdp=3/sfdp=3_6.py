
class Model(torch.nn.Module):
    def __init__(self, scale_factor=1.0, dropout_p=0.0):
        super().__init__()
        self.dropout_p = dropout_p
        self.scale_factor = scale_factor

    def forward(self, queries, keys, values):
        qk = queries.matmul(keys.transpose(-2, -1))
        scaled_qk = qk * self.scale_factor
        softmax_qk = torch.nn.functional.softmax(scaled_qk.mul(-1), dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(values)
        return output

# Initializing the model
m = Model(1.0, 0.0)

# Inputs to the model
query = torch.randn(1, 8, 16)
key = torch.randn(1, 8, 100)
value = torch.randn(1, 8, 100)
