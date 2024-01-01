
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout_p = 0.5
 
    def forward(self, query, key, value):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scale_factor = 1.0 / math.sqrt(qk.size(-1))
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = torch.nn.functional.dropout(scaled_qk.softmax(dim=-1), p=self.dropout_p)
        output = softmax_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
m.dropout_p=0.2
query = torch.randn(1, 2, 4)
