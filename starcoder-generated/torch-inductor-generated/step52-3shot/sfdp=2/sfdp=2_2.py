
class Model(torch.nn.Module):
    def __init__(self, query, key, value, bias, scale_factor, dropout_p):
        super().__init__()
        self.matmul1 = torch.nn.MatMul('abc,adc->abd')
        self.matmul2 = torch.nn.MatMul('abc,adc->abd')
 
    def forward(self, query_t, key_t, value_t, bias_t):
        qk = self.matmul1(query_t, key_t.transpose(1, 2))
        scale_qk = qk / scale_factor
        softmax_qk = torch.nn.functional.softmax(scale_qk, -1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p, training=True)
        output = self.matmul2(dropout_qk, value_t)
        return output

# Initializing the model
query_t = torch.ones(1, 20, 75)
key_t = torch.ones(1, 50, 75)
value_t = torch.ones(1, 50, 100)
bias_t = torch.randn(1, 1, 1, 100)
__model__ = nn.Model(query_t, key_t, value_t, bias_t, scale_factor=1.0, dropout_p=0.2)

# Inputs to the model
__inputs__ = [query_t, key_t, value_t, bias_t]
