
class Model(torch.nn.Module):
    def __init__(self, query, key, value, inv_scale_factor, dropout_p, dropout_input):
        super().__init__(query, key, value, inv_scale_factor, dropout_p, dropout_input)
        self.query = query
        self.key = key
        self.value = value
        self.inv_scale_factor = inv_scale_factor
        self.dropout_p = dropout_p
        self.dropout_input = dropout_input
 
    def forward(self):
        qk = torch.matmul(self.query, self.key.transpose(-2, -1))
        scaled_qk = qk.div(self.inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, self.dropout_p)
        output = dropout_qk.matmul(self.value)
        return output
 
# Initializing the model
query = torch.randn(1024, 512)
key = torch.randn(1024, 512)
value = torch.randn(1024, 512)
inv_scale_factor = 1.0 / math.sqrt(query.size(1))
dropout_p = 0.2
dropout_input = False
m = Model(query, key, value, inv_scale_factor, dropout_p, dropout_input)
 
# Inputs and output to the model
__input__ = (query, key, value, inv_scale_factor, dropout_p, dropout_input)
