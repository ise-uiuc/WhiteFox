
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(p)

    def forward(self, q, k, v, scale):
        scaled_qk = torch.matmul(q, k.transpose(-2, -1)) / scale
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = self.dropout(softmax_qk)
        output = (dropout_qk.matmul(v))
        return output

