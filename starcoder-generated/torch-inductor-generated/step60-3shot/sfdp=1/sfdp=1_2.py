
class Model(torch.nn.Module):
    def __init__(self, num_hidden=128):
        super().__init__()
        self.key_conv = torch.nn.Conv1d(num_hidden, num_hidden, 1, stride=1, padding=1)
 
    def forward(self, query, key, value):
        k = self.key_conv(key)
        inv_scale_factor = 1.0 / math.sqrt(k.size(-1))
        qk = torch.matmul(query, k.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.3)
        output = dropout_qk.matmul(value)
        return output
m = Model()
# Inputs to the model
query = torch.randn(1, 128, 1)
value = torch.randn(1, 128, 10)
key = torch.randn(1, 128, 5)
