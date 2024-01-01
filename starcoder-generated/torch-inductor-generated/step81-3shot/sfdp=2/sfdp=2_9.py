
class Model(torch.nn.Module):
    def forward(self, query, key, value, dropout_p=0.):
        inv_scale_factor = np.sqrt(query.shape[-1])
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(
            softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output
    
# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 8, 256)
key = torch.randn(1, 4, 256)
value = torch.randn(1, 4, 256)
dropout_p = 0.
