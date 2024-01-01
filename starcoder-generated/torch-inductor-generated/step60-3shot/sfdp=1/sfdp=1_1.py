
class Model(torch.nn.Module):
    def forward(self, __query__, __key__, __value__):
        qk = torch.matmul(__query__, __key__.transpose(-2, -1))
        scaled_qk = qk.div(__inv_scale_factor__)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=__dropout_p__)
