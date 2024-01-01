
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.rand((24, 24))   
 
    def forward(self, x1, x2):
        qk = torch.matmul(x1, x2.transpose(-2, -1))
        inv_scale_factor = torch.rsqrt(torch.float32.max).to(x1.device)
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.1)
        output = dropout_qk.matmul(self.weight)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(5, 24, 2)
x2 = torch.randn(5, 2, 24)
