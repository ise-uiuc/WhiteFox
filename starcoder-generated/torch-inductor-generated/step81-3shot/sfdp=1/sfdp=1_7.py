
class Model(torch.nn.Module):
    def forward(self, input_tensors, params):
        q = input_tensors['query']
        k = input_tensors['key']
        v = input_tensors['value']
        scale_factor = params['scale_factor']
        dropout_p = params['dropout_p']
        inv_scale_factor = 1.0 / scale_factor
        softmax_qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = softmax_qk * inv_scale_factor
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output_tensor = dropout_qk.matmul(v)
        return {'output_tensor': output_tensor}
 
# Initializing the model
m = Model()

# Inputs to the model
# query input tensor
q = torch.randn(1, 64, 1000) 
# key input tensor
k = torch.randn(1, 128, 1000) 
# value input tensor
v = torch.randn(1, 128, 1000) 
params = {
  'scale_factor': 128,
    'dropout_p': 0.1,
}
input_tensors = {'query': q, 'key': k, 'value': v}
output1 = m(input_tensors, params)

