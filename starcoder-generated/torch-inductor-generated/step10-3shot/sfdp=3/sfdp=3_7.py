
class Model(torch.nn.Module):
    def forward(self, x1, x2):
        qk = torch.matmul(x1, x2.transpose(-2, -1))
        scaled_qk = qk.mul(SCALE_FACTOR)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=DROPOUT_P)
        output = dropout_qk.matmul(x2)
        return output

# Randomly initialize the encoder layer
enc_layer = Model()

# Inputs to the encoder layer
x1 = torch.randn(1, 1, 512)
x2 = torch.randn(1, 1, 512)
