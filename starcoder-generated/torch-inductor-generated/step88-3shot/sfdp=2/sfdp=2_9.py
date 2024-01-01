
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(5, 200, 18, 18))
 
    def forward(self, x1):
        q = x1
        k = x1
        inv_scale_factor = 3200
        dropout_p = 0.2
        v = self.w
        qkv = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qkv.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 5, 18, 18)
