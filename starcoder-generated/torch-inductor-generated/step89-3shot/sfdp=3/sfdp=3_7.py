
class Model(torch.nn.Module):
    def forward(self, input_tensor, weight_tensor, bias):
        scale_factor = torch.tensor(2.0 ** 0.5, dtype=input_tensor.dtype, device=input_tensor.device)
        qk = torch.matmul(input_tensor[:, :5, :, :], weight_tensor.transpose(-2, -1))
        qk = qk.mul(scale_factor)
        qk = qk.softmax(dim=-1)
        qk = torch.nn.functional.dropout(qk, p=0.5)
        out = qk.matmul(weight_tensor)
        out = out + bias
        return out

# Initializing the model
weight_tensor = torch.randn(2, 5)
bias = torch.randn(2)
m = Model()
x1 = torch.randn(10, 7, 32, 32)
