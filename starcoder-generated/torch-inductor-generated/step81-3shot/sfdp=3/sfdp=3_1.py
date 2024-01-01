
class Model(torch.nn.Module):
    def __init__(self, num_heads, d_model, dropout_p, scale_factor):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.dropout_p = dropout_p
        self.scale_factor = scale_factor

    def attention(self, query, key, value):
        qk_mat = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk_mat = qk_mat.mul(self.scale_factor)

        softmax_qk_mat = scaled_qk_mat.softmax(dim=-1)
        dropout_qk_mat = torch.nn.functional.dropout(softmax_qk_mat, p=self.dropout_p)

        output = dropout_qk_mat.matmul(value)

        return output

    def forward(self, query, key, value):
        batch_size, q_len = query.size(0), query.size(1)
        head_size = q_len // self.num_heads
        if head_size * self.num_heads!= q_len:
            return None

        h = self.attention(query, key, value)
        return h

# Initializing the model
m = Model(num_heads=8, d_model=256, dropout_p=0.5, scale_factor=1)

# Inputs to the model
query = torch.randn(1, 256, 12, 64)
key = torch.randn(1, 256, 24, 64)
value = torch.randn(1, 256, 24, 64)
