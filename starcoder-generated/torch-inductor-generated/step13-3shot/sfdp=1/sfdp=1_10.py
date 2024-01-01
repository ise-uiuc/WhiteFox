
class Model(torch.nn.Module):
    def forward(self,query, key, value, inv_scale_factor=1.0, dropout_p=0.2):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        drop_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = drop_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1,512,17,17)
key = torch.randn(1,512,17,17)
value = torch.randn(1,512,17,17)
