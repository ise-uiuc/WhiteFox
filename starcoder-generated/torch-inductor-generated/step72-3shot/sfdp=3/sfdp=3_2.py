
class Model(torch.nn.Module):
    def __init__(self, scale_factor=0.5, dropout_p=0.):
        super().__init__()
        self.scale_factor = scale_factor
        self.dropout_p = dropout_p

    def forward(self, query, key, value):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.mul(self.scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dout = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = torch.matmul(dout, value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(16, 32, 64)
key = torch.randn(16, 32, 64)
value = torch.randn(16, 32, 64)
