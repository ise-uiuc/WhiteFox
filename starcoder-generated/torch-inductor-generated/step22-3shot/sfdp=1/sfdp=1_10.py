
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout_p = 0.0
 
    def forward(self, query, key, value, invscale_factor):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(invscale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout)
        output = dropout_qk.matmul(value)
        return output, dropout_qk
 
# Initializing the model
m = Model()

# Inputs to the model
shape = (1, 128, 5)
query = torch.randn(shape)
key = torch.randn(shape)
value = torch.randn(shape)
invscale_factor = torch.randn(shape[0], shape[1], 1)
__output__, __dropout_qk__ = m(query, key, value, invscale_factor)

