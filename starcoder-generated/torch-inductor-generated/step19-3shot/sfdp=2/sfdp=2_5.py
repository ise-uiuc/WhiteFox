
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1):
        q1 = x1 * 0.1
        k1 = x1 * 0.2
        v1 = x1 * 0.3
        dropout_p = 0.4
        inv_scale_factor = 512
 
        qk = q1 * k1.transpose(-2, -1)
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(v1)
 
        return output[-1]

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(10, 10, 10)
