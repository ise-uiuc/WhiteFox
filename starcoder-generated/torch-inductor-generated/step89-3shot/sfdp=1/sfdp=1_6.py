
class Model(torch.nn.Module):
    def __init__(self, query, key, value, scale_factor, dropout_p):
        super().__init__()
        self.query = query
        self.key = key
        self.value = value
        self.scale_factor = scale_factor
        self.dropout_p = dropout_p
 
    def forward(self):
        qk = torch.matmul(self.query, self.key.transpose(-2, -1))
        scaled_qk = qk.div(self.scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(self.value)
        return output

# Initializing the model
query = torch.randn(1, 8, 60, 256)
key = torch.randn(1, 8, 20, 256)
value = torch.randn(1, 8, 20, 256)
scale_factor = 10.0
dropout_p = 0.1
