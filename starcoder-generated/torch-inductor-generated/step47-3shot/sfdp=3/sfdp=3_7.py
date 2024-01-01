
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, query, key, value, **kwargs):
        qk = query @ key.transpose(-2, -1)
        scale_factor = kwargs.get("scale_factor", 1 / np.sqrt(query.size(-1)))
        scaled_qk = scale_factor * qk
        softmax_qk = F.softmax(scaled_qk, dim=-1)
        dropout_qk = F.dropout(softmax_qk, p=0.5, training=self.training)
        output = dropout_qk @ value
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(5, 8, 12)
key = torch.randn(5, 35, 12)
value = torch.randn(5, 35, 24)
