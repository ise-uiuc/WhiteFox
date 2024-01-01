
class Model(torch.nn.Module):
    def __init__(self, inv_scale_factor=1.0, dropout_p=0.1):
        super().__init__()
        self.scaled_dot_product_attention = torch.nn.Softmax(dim=-1)
        self.dropout = torch.nn.Dropout(dropout_p)
 
    def forward(self, query, key, value, inv_scale_factor, dropout_p):
        matmul1_qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = matmul1_qk.div(inv_scale_factor)
        softmax_qk = self.scaled_dot_product_attention(scaled_qk)
        dropout_output = self.dropout(softmax_qk)
        matmul2 = torch.matmul(dropout_output, value)
        return matmul2
    
# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(3, 2, 5)
key = torch.randn(5, 4, 5)
value = torch.randn(5, 4, 6)
