
class Model(torch.nn.Module):
    def __init__(self,
                 m,
                 n,
                 k,
                 dropout_p,
                 inv_scale_factor):
        super().__init__()
        self.q_fc = torch.nn.Linear(m, n)
        self.k_fc = torch.nn.Linear(k, n)
        self.dropout = torch.nn.Dropout(dropout_p)
        self.inv_scale_factor = inv_scale_factor
 
    def forward(self, q, k, v, input_mask=None):
        q = self.q_fc(q)
        k = self.k_fc(k)
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.div(self.inv_scale_factor)
        if input_mask:
            scaled_qk = scaled_qk.float().masked_fill(input_mask == 0, float('-inf')).type_as(scaled_qk)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = self.dropout(softmax_qk)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = Model(m, n, k, dropout_p, inv_scale_factor)

# Inputs to the model
q = torch.randn(1, n, m)
k = torch.randn(1, n, k)
v = torch.randn(1, n, k)
input_mask = torch.ones(1, 1, k)
