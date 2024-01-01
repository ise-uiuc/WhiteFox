
class Model(torch.nn.Module):
    def forward(self, q, k, v, padding_mask, dropout_p):
        qk = torch.matmul(q, k.transpose(-2, -1))
        scale_factor = math.sqrt(float(q.shape[-1]))
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(v)
        output = torch.where(padding_mask, query, output).long()
        return output
 
# Initializing the model
m = Model()
 
# Inputs to the model
query = torch.randn(1, 23, 768)
key = torch.randn(90, 23, 512)
value = torch.randn(90, 23, 512)
batch_size = 1
padding_mask = torch.zeros((batch_size, query.shape[1], key.shape[1]), dtype=torch.bool)
 
dropout_p = 0.1
