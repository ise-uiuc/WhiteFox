
import torch
import torch.nn.functional as F
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, q1, k1):
        qk = torch.matmul(q1, k1.transpose(-2, -1))
        inv_scale_factor = 1 / np.sqrt(q1.shape[-1])
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = F.dropout(softmax_qk, p=0.20000000298023224, training=self.training, inplace=False)
        output = dropout_qk.matmul(y1)
        return output

# Initializing the model
m = Model()

# Inputs to the model
q1 = torch.randn(1, 128, 8, device="cpu",dtype=torch.double)
k1 = torch.randn(1, 128, 16, device="cpu",dtype=torch.double)
__output__= m(q1, k1)

