
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(0.05)
 
    def forward(self, q, k, v):
        qk = torch.matmul(q, k.transpose(-2, -1))
        # compute qk, it is a tensor with shape (batch, n_head, q_len, k_len). Here the batch size is n_head * m_batch.
        k_dim = k.shape[-1]
        inv_scale_factor = math.sqrt(k_dim)
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        # compute softmax_qk, it is a tensor with shape (batch, n_head, q_len, k_len).
        dropout_qk = self.dropout(softmax_qk)
        # compute dropout_qk, it is a tensor with shape (batch, n_head, q_len, k_len).
        ouput = dropout_qk.matmul(v)
        # compute ouput, it is a tensor with shape (batch, n_head, q_len, v_dim). Here the batch size is n_head * m_batch.
        return output
 
# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(12, 9, 4, 10)
k = torch.randn(12, 9, 10, 4)
v = torch.randn(12, 9, 10, 20)
