
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, key, value, dropout_p, inv_scale_factor, training):
      kq = torch.matmul(query, key.transpose(-2, -1))
      scale_factor = kq.size(-1) ** 0.25
      scale_factor = scale_factor.to(query.device)
      scale_factor = scale_factor.to(query.dtype)
      scaled_kq = qk.div(scale_factor)
      softmax_kq = scaled_kq.softmax(dim=-1)
      dropout_kq = F.dropout2d(softmax_kq, p=dropout_p, training=training)
      output = dropout_kq.matmul(value)
      return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 3, 64, 64)
key = torch.randn(1, 3, 64, 64)
value = torch.randn(1, 3, 64, 64)
dropout_p = 0.2
inv_scale_factor = 1 / math.sqrt(0.1)
