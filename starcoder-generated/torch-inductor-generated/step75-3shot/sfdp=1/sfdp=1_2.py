
class Model(torch.nn.Module):
    def forward(self, query, key, value):
        query = query.softmax(axis=-1)
        q_k = query.dot(key.T)
        scaled_q_k = q_k / math.sqrt(q_k.size(-1))
        softmax_q_k = torch.softmax(q_k, -1)
        dropout_q_k = torch.nn.functional.dropout(softmax_q_k, p=0.8)
        return dropout_q_k.dot(value)

# Initializing the model
m = Model()

# Inputs to the model
__query__ = np.random.rand(4, 5)
__key__ = np.random.rand(4, 3)
__value__ = np.random.rand(4, 3)
