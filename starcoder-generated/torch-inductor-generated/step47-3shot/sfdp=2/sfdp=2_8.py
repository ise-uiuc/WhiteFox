
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, q, k, v):
        qk = torch.matmul(q, k.transpose(-2, -1))
        qk = qk.div(inv_scale_factor)
        softmax_qk = torch.nn.functional.softmax(qk, dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(v)
        return output
 
