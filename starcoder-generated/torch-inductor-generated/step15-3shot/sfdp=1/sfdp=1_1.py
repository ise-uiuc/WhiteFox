
class Model(torch.nn.Module):
    def __init__(self, dim, dropout_p):
        super().__init__()
        self.dropout_p = dropout_p
        self.softmax = torch.nn.Softmax(dim=dim)
        self.dropout = torch.nn.Dropout(p=dropout_p)

    # The `inv_scale_factor` is a constant tensor that can be loaded from memory to calculate the scale factor from dot product of query and key.
    def forward(self, query, key, value, inv_scale_factor):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = self.softmax(scaled_qk)
        dropout_qk = self.dropout(softmax_qk)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model(dim, dropout_p)

# Input tensors to the model
query = torch.randn(1, tgt_len, embed_dim)
key = torch.randn(1, src_len, embed_dim)
value = torch.randn(1, src_len, embed_dim)
inv_scale_factor = torch.ones(1)
