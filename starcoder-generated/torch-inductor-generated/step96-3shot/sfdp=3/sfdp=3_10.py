
class Model(torch.nn.Module):
    def forward(self, q1, k1, v1, dropout_p):
        q2 = torch.matmul(q1, k1.transpose(-2, -1))
        scale_factor = np.sqrt(query_channels // heads)
        q3 = q2.mul(scale_factor)
        softmax_qk = q3.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        v2 = dropout_qk.matmul(v1)
        return v2

# Initializing the model
m = Model()

query = torch.randn(7, 5, 6)
key = torch.randn(7, 5, 6)
value = torch.randn(7, 5, 6)
