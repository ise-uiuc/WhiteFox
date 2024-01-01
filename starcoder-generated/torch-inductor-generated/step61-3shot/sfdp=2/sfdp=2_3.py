
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inv_scale_factor = torch.nn.Parameter(torch.tensor(1.0))
        self.dropout_p = 0.1
 
    def forward(self, xq, xk, xv):
        qk = torch.matmul(xq, xk.transpose(-2, -1))
        scaled_qk = qk.div(self.inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(xv)
        return output

# Initializing the model
m = Model()

# Inputs to the model
xq = torch.randn(3, 8, 32)
xk = torch.randn(3, 8, 32)
xv = torch.randn(3, 8, 32)
