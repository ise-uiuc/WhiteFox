
class ScaledDotProductAttention(nn.Module):
    def forward(self, query, key, value, mask=None, dropout=None):
        scale_factor = query.size(-1) ** 0.5 # Use this formula to compute the inverse scale factor

        # To calculate the scaled dot product, uncomment the following line
        # qk = torch.matmul(query, key.transpose(-2, -1))

        qk = query * key.sum(dim=1, keepdim=True)
        if scale_factor!= 1:
            # Use this formula to scale the dot product
            # scaled_qk = qk.div(scale_factor)

            scaled_qk = qk / scale_factor
        
        # To apply softmax, uncomment the following line
        # softmax_qk = scaled_qk.softmax(dim=-1)

        softmax_qk = torch.softmax(scaled_qk, dim=-1)
        if dropout is not None:
            # To apply dropout, uncomment the following line
            # dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout)

            dropout_qk = dropout(softmax_qk)
        else:
            dropout_qk = softmax_qk
        output = dropout_qk * value
        return output


class Model(torch.nn.Module):
    def __init__(self, dropout=0.5, dim=10):
        super().__init__()
        self.dropout = dropout
        self.dim = dim
        self.attentions = list(ScaledDotProductAttention() for _ in range(dim))
        self.convs = list(nn.Conv3d() for _ in range(dim))
        self.norms = list(nn.LayerNorm() for _ in range(dim))

    def forward(self, x):
        v = x
        for norm, conv, attention in zip(self.norms, self.convs, self.attentions):
            v = norm(v)
            v = conv(v)
            v = attention(v)
        return v

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 10, 32, 64, 64)
