
class Model(torch.nn.Module):
    def __init__(self, dropout_rate):
        super().__init__()
        self.scale_factor = np.power(dropout_rate, 0.5)

    def forward(self, query, key, value):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.mul(self.scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_rate)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model(dropout_rate=0.3)

# Inputs to the model
query = torch.rand(1, 20, 256)
key = torch.rand(1, 20, 256)
value = torch.rand(1, 20, 256)
