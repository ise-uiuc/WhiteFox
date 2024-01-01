
class Model(torch.nn.Module):
    def forward(self, query, key, value, inv_scale_factor=None, dropout_p=0.1):
        qk = torch.matmul(query, key.transpose(-2, -1))
        if inv_scale_factor is not None:
            qk = qk.div(inv_scale_factor)
        softmax_qk = qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output
# Input to the model for illustration. The sizes of the query, the key, and the value are all (5, 4096, 512) by default
query = torch.randn(5, 4096, 512)
key = torch.randn(5, 4096, 512)
value = torch.randn(5, 4096, 512)
output = m(query, key, value)

