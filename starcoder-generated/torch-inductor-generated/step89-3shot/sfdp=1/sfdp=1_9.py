
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.dropout_p = drop_ratio
        self.inv_scale_factor = np.sqrt(1 / (dropout_ratio * embed_size))
 
    def forward(self, query, key, value):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(self.inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(value)
        return output


# Initializing the model
m = Model(embed_size=128, dropout_ratio= drop_ratio)

# Inputs to the model
query = torch.randn(16, 128)
key = torch.randn(32, 128)
value = torch.randn(32, 128)
