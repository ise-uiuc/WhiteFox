
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, key, value, dropout_p, inv_scale_factor):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model with hyperparameters
m = Model()

# Inputs to the model
query = torch.randn(seq_len, batch_size, feature_size)
key = torch.randn(seq_len, batch_size, feature_size)
value = torch.randn(seq_len, batch_size, feature_size)
dropout_p = 0.8
inv_scale_factor = torch.tensor(1.41421356237).view(1, 1, 1)
