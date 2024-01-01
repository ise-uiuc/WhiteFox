
class Model(torch.nn.Module):
    def __init__(self, query_feature_len, key_feature_len)
        super().__init__()        

    def forward(self, query, key, value):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output


# Initializing the model
m = Model(16, 32)

# Inputs to the model
query = torch.randn(1, 8, 16)
key = torch.randn(1, 4, 32)
value = torch.randn(1, 4, 32)
