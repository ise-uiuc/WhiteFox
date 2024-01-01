
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x1, x2, x3):
        q = x1.mean(dim=1)
        k = x2.view(x2.shape[0],-1)
        v = x3
        qk = torch.matmul(q, k.transpose(-2, -1))
        inv_scale_factor = math.sqrt(q.shape[-1])
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        p = 0.01
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=p)
        output = dropout_qk.transpose(1,2).matmul(v).squeeze(-1)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(12,16, 8)
x2 = torch.randn(12,32,12)
x3 = torch.randn(12,12,16)
