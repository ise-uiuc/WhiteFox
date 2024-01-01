
class SoftmaxDotAttention:
    def __init__(self, query, key, value, dropout_p=0.1):
        self.dropout_p = dropout_p
        self.softmax = torch.nn.Softmax(dim=-1)
        self._query = query
        self._key = key
        self._value = value
 
    def forward(self, query, key, value):
        qk = torch.matmul(query, key.transpose(-2, -1))
        inv_scale_factor = 1. / np.sqrt(len(self._query.flatten().shape))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = self.softmax(scaled_qk)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(value)
        return output

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = SoftmaxDotAttention(*torch.randn(1, 32, 64, 64), dropout_p=0.1)
 
    def forward(self, x):
        v1 = self.attention(x, x, x)
        return v1

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 32, 64, 64)
