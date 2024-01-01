
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1):
        qk = torch.matmul(x1, x1.T)
        inv_scale_factor = torch.rsqrt(torch.tensor(2048.)).to(x1.device)
        qk = qk / inv_scale_factor
        softmax_qk = F.softmax(qk, dim=-1)
        dropout_qk = F.dropout(softmax_qk, 0.3)
        return dropout_qk.matmul(x1)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2048, 256)
