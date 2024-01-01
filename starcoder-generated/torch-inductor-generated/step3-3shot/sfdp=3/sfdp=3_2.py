
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(d_model, dim)
 
    def forward(self, query, key, value, padding_mask, scale_factor=None, dropout_p=0.0):
        q = self.linear(query).view(-1, np.prod(query.size()[1:]))
        k = self.linear(key).view(-1, np.prod(key.size()[1:]))
        v = self.linear(value).view(-1, np.prod(value.size()[1:]))
        qk = torch.matmul(q, k.transpose(-2, -1))
        if scale_factor:
            scaled_qk = qk.mul(scale_factor)
        softmax_qk = scaled_qk.softmax(-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, dropout_p)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
key = torch.randn(7, 7, d_model)
value = torch.randn(7, 7, d_model)
padding_mask = torch.abs(torch.randn((7, 1, 7)) > 0.5).to(torch.float32)
scale_factor = torch.tensor(1 / (d_model ** 0.5)).type_as(key)
dropout_p = 0.1
