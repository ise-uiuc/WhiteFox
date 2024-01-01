
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, input0, input1, dropout_p):
        query = input0
        key = input1
        inv_scale_factor = 1.0 / math.sqrt(query.size(-1))
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
dropout_p = random.random()
x1 = torch.randn(10, 32, 512)
x2 = torch.randn(10, 32, 512)
