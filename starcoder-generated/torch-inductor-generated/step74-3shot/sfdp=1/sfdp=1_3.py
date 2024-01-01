
class Model(torch.nn.Module):
    # Note that here, the input of the model contains both the query and the key tensors.
    def forward(self, __query__, __key__, __value__, scale_factor, dropout_p): 
        qk = __query__.matmul(__key__.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(__value__)
        return output

# Initializing the model
m = Model()   

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 64, 64)
