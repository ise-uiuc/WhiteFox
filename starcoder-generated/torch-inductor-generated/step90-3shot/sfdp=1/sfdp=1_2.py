
class ScaledDotProductAttention(torch.nn.Module):
    def __init__(self, dropout_p):
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout_p)
 
    def forward(self, query, key, value, inv_scale_factor=None):
        qk = torch.matmul(query, key.transpose(-2, -1))
        if inv_scale_factor is not None:
            qk = qk.div(inv_scale_factor)
        softmax_qk = qk.softmax(dim=-1)
        dropout_qk = self.dropout(softmax_qk)
        p1 = dropout_qk.matmul(value)
        return p1, softmax_qk

# Inputs to the model
att = ScaledDotProductAttention(dropout_p=0.25)
query = torch.randn(1, 4, 20)
p2 = m(query)   
