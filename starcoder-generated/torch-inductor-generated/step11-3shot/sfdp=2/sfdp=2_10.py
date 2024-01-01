
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout_p = 0.2
 
    def forward(self, v1):
        qk = torch.matmul(v1, v1.transpose(-2, -1))
        inv_scale_factor = qk.size()[-1] ** -0.5
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(v1)
        return output

# Initializing the model
m = Model()

# Inputs to the model
v1 = torch.randn(1, 32, 5)
