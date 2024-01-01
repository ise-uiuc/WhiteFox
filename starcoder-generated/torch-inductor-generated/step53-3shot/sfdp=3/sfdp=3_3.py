
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale_factor = 1 / (np.power(2.0, 1 / 64) ** 2)

    def forward(self, query, key, value, dropout_p):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.mul(self.scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 64, 48)
key = torch.randn(1, 64, 64)
value = torch.randn(1, 64, 64)
dropout_p = 0.0
