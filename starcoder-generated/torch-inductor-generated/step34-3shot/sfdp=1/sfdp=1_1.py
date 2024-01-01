
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1):
        qk = x1.transpose(1, 2).matmul(x1)
        inv_scale_factor = (5000. / qk.size(1)) **.5
        scaled_qk = qk / inv_scale_factor
        softmax_qk = torch.nn.functional.softmax(scaled_qk, dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.1)
        out = dropout_qk.matmul(x1).transpose(1, 2)
        return out
 
# Initializing the model
m = Model()
 
# Input to the model
x1 = torch.randn(1, 64, 50)
