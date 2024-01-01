
class Model(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = input_dim * 16
        self.dropout_p = 0.1
        self.scale_factor = self.output_dim ** 0.5

    def forward(self, query, key, value):
        shape_qk = query.shape[:-1] + (self.output_dim, self.input_dim)
        shape_sk = key.shape[:-2] + shape_qk[-1:]
        shape_sv = value.shape[:-2] + shape_qk[-1:]

        qk = torch.matmul(query, key.transpose(-2, -1))
        qk = qk.view(shape_qk)
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value.view(shape_sv))
        return output, softmax_qk

# Initializing the model        
m = Model(768)

# Inputs to the model
query = torch.randn(1, 16, 768)
key = torch.randn(1, 20, 768)
value = torch.randn(1, 20, 768)
__o1__, __o2__ = m(query, key, value)

