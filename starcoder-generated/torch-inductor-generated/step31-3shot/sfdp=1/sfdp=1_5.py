
class Model(torch.nn.Module):
    def __init__(self, query, key, value, inv_scale_factor, dropout_p):
        super().__init__()
        self.query = query
        self.key = key
        self.value = value
        self.inv_scale_factor = inv_scale_factor
        self.dropout_p = dropout_p
 
    def forward(self):
        qk = torch.matmul(self.query, self.key.transpose(-2, -1))
        scaled_qk = qk.div(self.inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(self.value)
        return output

# Initializing the model parameters
query = torch.randn(2, 2, 2)
key = torch.randn(2, 1, 2)
value = torch.randn(2, 3, 2)
inv_scale_factor = 2.4
dropout_p = 0.03

# Initializing the model
m = Model(query, key, value, inv_scale_factor, dropout_p)

# Inputs to the model
