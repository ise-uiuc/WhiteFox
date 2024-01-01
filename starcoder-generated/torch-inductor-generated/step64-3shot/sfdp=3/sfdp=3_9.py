
class Model(torch.nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout_p):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_q = torch.nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_k = torch.nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_v = torch.nn.Linear(d_model, n_head * d_v, bias=False)

        self.attn_fc = torch.nn.Linear(n_head * d_v, d_model, bias=False)

        self.dropout_p = dropout_p
        self.dropout = torch.nn.Dropout(dropout_p)

    def forward(self, q, k, v):
        batch_size = q.size(0)
        q = self.w_q(q).view(batch_size, -1, self.n_head, self.d_k)
        k = self.w_k(k).view(batch_size, -1, self.n_head, self.d_k)
        v = self.w_v(v).view(batch_size, -1, self.n_head, self.d_v)

        q = q.permute(0, 2, 1, 3).contiguous().view(-1, len_q, self.d_k)
        k = k.permute(0, 2, 3, 1).contiguous().view(-1, len_k, self.d_k)
        v = v.permute(0, 2, 1, 3).contiguous().view(-1, len_v, self.d_v)

        scale_factor = (len_k ** -0.5)
        scaled_dot_product = torch.matmul(q, k.transpose(1, 2)) * scale_factor

        softmax_product = F.softmax(scaled_dot_product, dim=-1)
        dropout_product = self.dropout(softmax_product)

        linear_output = torch.matmul(dropout_product, v)
        sub_output = linear_output.view(batch_size, self.n_head, len_q, self.d_v)
        sub_output = sub_output.permute(0, 2, 1, 3).contiguous().view(batch_size, len_q, -1)

        attn_output = self.attn_fc(sub_output)

        return attn_output

# Initializing the model
d_model, d_k, d_v, n_head, len_q, len_k, len_v = 128, 16, 16, 8, 20, 10, 15
dropout_p = 0.1
m = Model(n_head, d_model, d_k, d_v, dropout_p)

# Inputs to the model
q = torch.randn(4, len_q, d_model)
k = torch.randn(4, len_k, d_model)
v = torch.randn(4, len_v, d_model)
