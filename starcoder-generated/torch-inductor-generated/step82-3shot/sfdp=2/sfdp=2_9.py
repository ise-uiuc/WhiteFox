
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=0.1)
 
    def forward(self, q, k, v, inv_scale_factor=1):
        qk = torch.matmul(q.float(), k.transpose(-2, -1).float())
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = self.dropout(softmax_qk)
        output = dropout_qk.matmul(v)
        return output
    
# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(1, 8, 64)
k = torch.randn(1, 8, 128)
v = torch.randn(1, 8, 128)
__out1__, __out2__, __out3__ = m(q, k, v)

