
class Model(torch.nn.Module):
    def __init__(self, n_heads, d_model, p=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.p = p

        self.scale_factor = np.sqrt(d_model)

    def forward(self, x1, x2):
        x1_reshape = x1.reshape((x1.shape[0], x1.shape[1], self.n_heads, -1))
        x1_transpose = x1_reshape.transpose(1, 2)
        x1_transpose = x1_transpose.reshape(
            (x1.shape[0], self.n_heads, x2.shape[1], -1)
        )
        x1_transpose = x1_transpose.transpose(-1, -2)
        x1_transpose = x1_transpose.reshape((x1.shape[0], x2.shape[1], -1))

        x2_reshape = x2.reshape((x2.shape[0], x1.shape[1], self.n_heads, -1))
        x2_transpose = x2_reshape.transpose(1, 2)
        x2_transpose = x2_transpose.reshape(
            (x1.shape[0], self.n_heads, x1.shape[1], -1)
        )
        x2_transpose = x2_transpose.transpose(-1, -2)
        x2_transpose = x2_transpose.reshape((x1.shape[0], x1.shape[1], -1))

        q = x1
        k = x2
        v = x2

        qk = torch.matmul(q, k.transpose(-2, -1))
        inv_scale_factor = torch.full_like(qk, -1 * self.scale_factor)
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.p)
        output = dropout_qk.matmul(v)
        output = output.transpose(-1, -2).reshape((q.shape[0], 1, -1, x2.shape[-1]))
        return output

# Initializing the model
m = Model(n_heads=8,
          d_model=32,
          )

# Inputs to the model
x1 = torch.randn(1, 64, 32)
x2 = torch.randn(1, 64, 32)
