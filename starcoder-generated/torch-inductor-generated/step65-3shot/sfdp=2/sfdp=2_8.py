
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = torch.nn.Softmax(dim=-1)
 
    def forward(self, x1):
        qk = torch.matmul(x1, x1.transpose(-2, -1))
        inv_scale_factor = torch.rsqrt((torch.mean(qk, -1)+1e-5).unsqueeze(-1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = self.softmax(scaled_qk)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.2)
        output = dropout_qk.matmul(kq)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 1, 200, 256)
