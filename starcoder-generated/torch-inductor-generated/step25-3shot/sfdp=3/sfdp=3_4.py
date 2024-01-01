
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward_helper(self, query_features, key_features, scale_factor, dropout_p):
        qk = torch.matmul(query_features, key_features.transpose(-2, -1))
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

    def forward(self, q, k, v, dropout_p=0.0):
        if q.shape[2] % k.shape[2] == 0:
            scale_factor = math.sqrt(k.shape[2])
        else:
            scale_factor = np.sqrt(k.shape[2])
        return self.forward_helper(q/scale_factor, k, scale_factor, dropout_p)

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(1, 23, 10000)
k = torch.randn(1, 768, 10000)
v = torch.randn(1, 768, 10000)
