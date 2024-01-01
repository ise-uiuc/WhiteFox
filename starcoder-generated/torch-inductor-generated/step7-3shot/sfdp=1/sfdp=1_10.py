
class Model(torch.nn.Module):
    def __init__(self, query, key, value, scale_factor, dropout_p):
            super().__init__()
            self.query = query
            self.key = key
            self.query = query
            self.dropout_p = dropout_p
            self.scale_factor = scale_factor
 
    def forward(self, query, key, value):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk / scale_factor
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
query = torch.randn(1, 3, 64, 64)
key = torch.randn(1, 3, 64, 64)
value = torch.randn(1, 3, 64, 64)
scale_factor = 1./sqrt(3.)
dropout_p = 0.2
