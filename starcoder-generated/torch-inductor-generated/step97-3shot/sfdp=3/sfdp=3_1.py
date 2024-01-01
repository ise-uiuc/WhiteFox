
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q = TensorList()
        self.k = TensorList()
        self.scale_factor = Parameter(torch.scalar_tensor(0.244444))
        self.dropout_p = 0.1

    def forward(self, query_input, key_input, value_input):
        q = self.q[input]
        k = self.k[input]
        qk = torch.matmul(query, key.transpose(-2, -1))
        scale_factor = self.scale_factor
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()
m.q.put(query_input)
m.k.put(key_input)
m.k.put(value_input)

