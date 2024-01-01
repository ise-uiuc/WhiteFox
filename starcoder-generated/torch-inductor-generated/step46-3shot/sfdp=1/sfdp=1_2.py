
class T5Attention(nn.Module):
    def __init__(self, q, k, v, mask, dropout_p=0.2):
        super().__init__()

        self.scale_factor = math.sqrt(q.size(-1))
        self.dropout = nn.Dropout(dropout_p)

        # Convert the query and key to float.
        q = q.float()
        k = k.float()

        # Apply dropout to the query and key tensor.
        self.q = self.dropout(q)
        self.k = self.dropout(k)

        # Generate a scale factor tensor and apply dropout to it.
        self.scale_factor = self.dropout(torch.tensor([self.scale_factor]))

        # Apply dropout to the value tensor.
        self.v = self.dropout(v)

        self.mask = mask

    def forward(self, x):
        # Compute the dot product of the query tensor and the key tensor.
        qk = torch.matmul(x, self.k.transpose(-2, -1))

        # Scale the dot product.
        scaled_qk = qk.div(self.scale_factor)

        # Apply softmax to the scaled dot product.
        softmax_qk = scaled_qk.softmax(dim=-1)

        # Apply dropout to the softmax output.
        dropout_qk = self.dropout(softmax_qk)

        # Apply the masking operation to the dropout output.
        masked_softmax_qk = softmax_qk.masked_fill(self.mask, float('-inf'))

        # Compute the dot product of the dropout output tensor and the value tensor.
        output = torch.matmul(masked_softmax_qk, self.v)

        return output

# Parameters
q = torch.randn(2, 3, 512, 1, dtype=torch.float)
k = torch.randn(2, 3, 1, 512, dtype=torch.float)
v = torch.randn(2, 3, 512, 512, dtype=torch.float)
mask = torch.randint(0, 2, (2, 1, 1, 512))

# Forward pass
attention = T5Attention(q, k, v, mask)
