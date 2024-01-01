
class Model(torch.nn.Module):
    def __init__():
        super().__init__()
 
    def forward(self, query, key, value, key_padding_mask, need_weights, attn_mask, return_attn_mask):
        x1 = torch.matmul(query, key.transpose(-2, -1))
        x2 = x1 / self.inverse_scale_factor
        x3 = torch.nn.functional.softmax(x2, dim=-1)
        x4 = torch.nn.functional.dropout(x3, p=self.dropout_p, training=self.training)
        x5 = torch.matmul(x4, value)
        return x5
 
# Initializing the model
model = Model(
  dropout_p=0.1,
  inverse_scale_factor=1000
  )
 
# Inputs to the model
input_query = torch.rand(2, 32, 32)
input_key = torch.rand(2, 32, 32)
input_value = torch.rand(2, 32, 32)
key_padding_mask = torch.rand(2, 32)
need_weights = False
attn_mask = torch.rand(32, 32)
return_attn_mask = False
