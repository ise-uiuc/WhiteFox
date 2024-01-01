
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, q, k, v, n_head=1, dropout_p=0.5):
        scale_factor = torch.sqrt(q.size(-1))

        q_copy = q
        k_copy = k
        v_copy = v

        q = self.linear_q(q).unsqueeze(0).transpose(0, 1)
        k = k_copy.unsqueeze(0).transpose(0, 1)
        v = v_copy.unsqueeze(0).transpose(0, 1)

        mask = torch.zeros((1, 1,) + (q.size(-1),))

        if self.training:
            mask = mask.bernoulli_(1 - dropout_p)

        mask = mask.expand((n_head, -1) + (-1,))

        q *= mask
        k *= mask
        v *= mask

        scaled_qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = scaled_qk.div(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = torch.matmul(dropout_qk, v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(1, 768)
k = torch.randn(1, 768)
v = torch.randn(1, 768)
