
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(0.2)
 
    def forward(self, query, key, value, inv_scale_factor, dropout_p=0.2):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = self.dropout(softmax_qk)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 2, 256, 64)
key = torch.randn(1, 2, 64, 256)
value = torch.randn(1, 2, 256, 256)
inv_scale_factor = 1. / torch.sqrt(torch.tensor(query.shape[-1], dtype=torch.float))
x5 = m(query, key, value, inv_scale_factor)

