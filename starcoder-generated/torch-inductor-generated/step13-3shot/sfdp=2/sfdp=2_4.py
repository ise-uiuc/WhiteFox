
class Model(torch.nn.Module):
    def __init__(self, inv_scale_factor, dropout_p):
        super().__init__()
        self.dropout_p = dropout_p
        self.inv_scale_factor = inv_scale_factor
 
    def forward(self, query, key, value):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(self.inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
inv_scale_factor = 4
dropout_p = 0.005
m = Model(inv_scale_factor, dropout_p)

# Inputs to the model
query = torch.randn(1, 5, 3)
__key__ = torch.randn(2, 5, 3)
__value__ = torch.randn(2, 5, 7)
output = m(query, __key__, __value__)

