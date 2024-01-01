
class Model(torch.nn.Module):
    def __init__(self):
        pass
 
    def forward(self, *args):
        query, key, value, scale_factor, dropout_p = args
        qk = torch.matmul(query, key.transpose(-2, -1)) 
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 6, 4, 5)
key = torch.randn(5, 4, 1)
value = torch.randn(5, 4, 32)
scale_factor = 0.2
dropout_p = 0.8
__output__   = m(query, key, value, scale_factor, dropout_p)
output_list = [x.shape for x in __output__.values()]  ## You can also use torchsummaryx. summary(m) to retrieve output shape.

