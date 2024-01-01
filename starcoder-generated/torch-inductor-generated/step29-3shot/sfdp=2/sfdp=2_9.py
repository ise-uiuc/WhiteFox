
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout2d(0.2)
        self.ln = torch.nn.LayerNorm([16, 64, 64])

    def forward(self, x1):
        # First layer
        q = x1
        k = x1
        v = x1

        # Second layer
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = self.dropout(softmax_qk)
        output = output = dropout_qk.matmul(v)

        return self.ln(output)

        