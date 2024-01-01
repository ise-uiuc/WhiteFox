
class Model(torch.nn.Module):
    def __init__(self, input_x, input_y, input_z):
        super().__init__()
 
    def forward(self, q, k, v):
        qk = torch.matmul(q, k.transpose(-2, -1))
        inv_scale_factor = math.sqrt(q.size(-1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.1)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
input_x = torch.randn(1, 2, 8, 8)
input_y = torch.randn(1, 4, 8, 8)
input_z = torch.randn(3, 4, 1, 1)
