
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
       
    def forward(self, x1, x2):
        qk = torch.matmul(x1, x2.transpose(-2, -1))
        scaled_qk = qk.div(self.inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(8, 1024, 32)
x2 = torch.randn(8, 32, 32)
