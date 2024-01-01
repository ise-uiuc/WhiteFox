
import torch

class Model(torch.nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.hidden_dims = (input_shape[0]*input_shape[1]*input_shape[2], 8*input_shape[2]*input_shape[3])
        self.query = torch.nn.Linear(*self.hidden_dims)
        self.key = torch.nn.Linear(*self.hidden_dims)
        self.value = torch.nn.Linear(*self.hidden_dims)

    def forward(self, x1):
        qk = self.query(x1).matmul(self.key(x1).transpose(-2, -1))
        in_scale_factor = np.sqrt(q.size(-1))
        softmax_qk = (qk / in_scale_factor).softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, training=True)
        self.dropout_qk = dropout_qk
        output = self.dropout_qk.matmul(self.value(x1))
        return output

# Initializing the model
m = Model((3, 16, 64, 64))

# Inputs to the model
x1 = torch.randn(3, 16, 64, 64)
