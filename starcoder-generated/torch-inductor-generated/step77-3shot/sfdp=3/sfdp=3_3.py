
class Model(torch.nn.Module):
    def __init__(self, q, k, v): # Inputs to the model
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.mul(scale_factor)
        self.softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        self.output = dropout_qk.matmul(v)
 
    def forward(self):
        return self.output
 
def sample_model_inputs(input_shape=(1, 1, 1)):
    input = np.random.random_sample(input_shape)
    return input
 
q = torch.tensor(sample_model_inputs((16, 8, 16)))
k = torch.tensor(sample_model_inputs((16, 16, 8)))
v = torch.tensor(sample_model_inputs((16, 8, 8)))
 
