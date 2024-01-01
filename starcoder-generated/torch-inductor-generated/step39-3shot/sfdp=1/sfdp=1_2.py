
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout_p = p # Make p a class member
 
    def forward(self, q1, k1, v1, inv_scale_factor):
        qk = torch.matmul(q1, k1.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(v1)
        return output

# Initializing the model
m = Model()

# Inputs to the model
q1 = torch.randn(8, 64, 256)
k1 = torch.randn(8, 256, 416)
v1 = torch.randn(8, 416, 256)
inv_scale_factor = 0.125
