
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, key, value, inv_scale_factor, dropout_p, attention_mask):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        assert list(output.shape) == attention_mask.shape, '{} vs {}'.format(output.shape, attention_mask.shape)
        return output * attention_mask

# Initializing the model
n = Model()

# Inputs to the model
query = torch.randn(32, 196, 1536)
key = torch.randn(32, 2048, 64)
value = torch.randn(32, 2048, 64)
inv_scale_factor = 1 / math.sqrt(64)
dropout_p = 0.5
attention_mask = torch.zeros(32, 196, 2048, dtype=torch.float32)
attention_mask[:, :, 128:256] = 1
