
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()        
        self.inv_scale_factor = 1.06
        self.dropout_p = 0.5

    def forward(self, query, key, value, x1):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk / self.inv_scale_factor
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)        
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 768, 784)
key = torch.randn(1, 784, 784)
value = torch.randn(1, 768, 768)

x1 = torch.randn(1, 768, 768)
