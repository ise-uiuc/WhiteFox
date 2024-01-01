
class Model(torch.nn.Module):
    def forward(self, query_, key_, value_, input_mask_, dropout_p):
        Q = self.query_weight(query_)
        K = self.key_weight(key_)
        V = self.value_weight(value_)
        QK = torch.matmul(Q, K)
        inv_scale_factor = torch.tensor(Q.shape[-1]).type_as(QK)
        QK = QK.div(inv_scale_factor)
        QK = softmax(QK, -1)
        QK = dropout(QK, dropout_p)
        V = dropout(V, dropout_p)
        Y = torch.matmul(QK, V)
        return Y

# Initializing the model
m = Model()

# Inputs to the model
query_ = torch.randn(1, 2, 16, 16)
key_ = torch.randn(1, 2, 16, 16)
value_ = torch.randn(1, 3, 16, 16)
input_mask_ = torch.ones(1, 64, 64)
dropout_p = 0.1
__output__, __dpu_op__ = m(query_, key_, value_, input_mask_, dropout_p)

