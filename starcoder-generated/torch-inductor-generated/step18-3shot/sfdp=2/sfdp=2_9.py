
class Model(torch.nn.Module):
    def __init__(self, inv_scale_factor, dropout_p):
        super().__init__()
        self.inv_scale_factor = inv_scale_factor
        self.dropout_p = dropout_p
 
    def forward(self, x1, x2):
        qk = torch.matmul(x1, x2.transpose(-2, -1))
        scaled_qk = qk.div(self.inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, self.dropout_p)
        output = dropout_qk.matmul(x2)
        return output 

# Initializing the model
m = Model(1, 0.5)

# Inputs to the model
x1 = torch.randn(16, 128, 24)
x2 = torch.randn(16, 24, 128)
