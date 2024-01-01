
class Model(torch.nn.Module):
    def __init__(self, scale_factor=1, dropout_p=0.1):
        super().__init__()
        
        # Note that this pattern is not specific to transformer models.
        # This is just a placeholder to demonstrate the same pattern can be applied to other models.
        self.scale_factor = scale_factor
        self.dropout_p = dropout_p

    def forward(self, query, key, value):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.mul(self.scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(value)
        return output
 
# Initializing the model
scale_factor = 1.0
dropout_p = 0.1
m = Model(scale_factor, dropout_p)

# Inputs to the model
query = torch.rand(3, 115, 196)
key = torch.rand(3, 230, 196)
value = torch.rand(3, 230, 196)
