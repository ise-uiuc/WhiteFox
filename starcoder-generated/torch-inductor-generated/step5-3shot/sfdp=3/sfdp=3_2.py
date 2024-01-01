
def scaled_dot_product_attention(query, key, value, scale_factor=0.1, dropout_p=0.3):
    qk = torch.matmul(query, key.transpose(-2, -1))
    scaled_qk = qk.mul(scale_factor)
    softmax_qk = scaled_qk.softmax(dim=-1)
    dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
    output = dropout_qk.matmul(value)
    return output
 
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query = torch.nn.Linear(3, 6)
        self.key = torch.nn.Linear(3, 6)
        self.value = torch.nn.Linear(3, 6)
 
    def forward(self, input):
        query = self.query(input)
        key = self.value(input)
        value = self.value(input)
        output = scaled_dot_product_attention(query, key, value)
        return output
       
# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
