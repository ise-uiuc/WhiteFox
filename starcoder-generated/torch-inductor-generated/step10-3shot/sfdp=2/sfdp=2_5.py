
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, query, key, value, inv_scale_factor, dropout_p):
        qk = query.bmm(key.transpose(1,2))
        scaled_qk = (qk / inv_scale_factor.view(qk.shape[0], qk.shape[1], 1)).softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(scaled_qk, p=dropout_p)
        output = dropout_qk.bmm(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 12, 512)
key = torch.randn(1, 12, 512)
value = torch.randn(1, 12, 512)
inv_scale_factor = torch.randn(12)
dropout_p = 0.3
