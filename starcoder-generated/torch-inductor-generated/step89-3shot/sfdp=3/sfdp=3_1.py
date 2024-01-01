
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = 1

    def forward(self, query, key, value):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.mul(self.scale)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()
m.scale = 2

# Inputs to the model
query = torch.randn(1, 2, 4, 4)
key = torch.randn(1, 2, 4, 8)
value = torch.randn(1, 2, 4, 8)
