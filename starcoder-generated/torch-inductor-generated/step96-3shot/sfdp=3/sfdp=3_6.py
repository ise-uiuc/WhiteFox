
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout_p = 0
        self.dropout = torch.nn.Dropout(p=0)
 
    def forward(self, __input_query__, __input_key__, __input_value__):
        qk = torch.matmul(__input_query__, __input_key__.transpose(-2, -1))
        scaled_qk = qk.mul(1)
        softmax_qk = torch.nn.functional.softmax(scaled_qk, dim=-1)
        dropout_qk = self.dropout(softmax_qk)
        output = dropout_qk.matmul(__input_value__)
        return output


# Initializing the model
m = Model()

# Inputs to the model
__input_query__ = torch.randn(64, 128, 64)
__input_key__ = torch.randn(64, 128, 64)
__input_value__ = torch.randn(64, 128, 128)
