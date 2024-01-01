
class BertModel(torch.nn.Module):
    def __init__(self):
        super(BertModel, self).__init__()
        self.w_aout = Linear(4, 4)
    def forward(self, q, k, v, wq, wk):
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        wk = w_aout(output)  # shape (batch, nheads, sequence, sequence)

# Initializing the model
m = BertModel()

# Input to the model
q = torch.einsum("nhd,hdk->nhdk", q, self.w_q(q))
k = torch.einsum("nhd,hdk->nhdk", k, self.w_k(k))  # (batch_size, nheads, 2*sequence_length, dim_per_head)
v = torch.einsum("nhd,hdk->nhdk", v, self.w_v(v))
