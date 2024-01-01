
class Model(torch.nn.Module):
    def __init__(self, scale_factor, dropout_p):
        super().__init__()
        self.scale_factor = scale_factor
        self.dropout_p = dropout_p
 
    def forward(self, query, key, value):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(self.scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model(0.1, 0.5)

# Inputs to the model
query = torch.randn(12, 16, 256)
key = torch.randn(12, 16, 256)
value = torch.randn(12, 16, 256)
