
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
 
    def forward(self, query, key, value, scale_factor, dropout_p):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(scale_factor)
        output = torch.nn.functional.dropout(torch.nn.functional.softmax(scaled_qk, dim=-1), p=dropout_p).matmul(value)
        return output
 
# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(5, 16, 128)
key = torch.randn(5, 128, 256)
value = torch.randn(5, 128, 256)
scale_factor = (query.size(0) * query.size(1)) ** -(0.25)
dropout_p = 0.25
