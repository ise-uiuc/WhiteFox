
class Model(torch.nn.Module):
    def __init__(self, query, key, value, dropout_p, inv_scale_factor):
        super().__init__()
        self.query = torch.nn.Parameter(query)
        self.key = torch.nn.Parameter(key)
        self.value = torch.nn.Parameter(value)
        self.dropout_p = dropout_p
        self.inv_scale_factor = inv_scale_factor
 
    def forward(self, scaled_query):
        qk = torch.matmul(scaled_query, self.key.transpose(-2, -1))
        softmax_qk = qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(self.value)
        return output

# Values of trainable parameters in the model
query = torch.randn(1, 13, 768)
key = torch.randn(1, 13, 768)
value = torch.randn(1, 13, 768)
dropout_p = 0.2
inv_scale_factor = 1 / sqrt(768)

# Inputs to the model
scaled_query = torch.randn(1, 6, 768)
