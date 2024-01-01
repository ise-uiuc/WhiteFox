
class Model(torch.nn.Module):
    def __init__(self, batchsize=1, n_head=1,
                 query_channels=128, key_channels=128, value_channels=128,
                 input_tensor_shape=(16, 3, 128, 128), scale_factor=1.0/sqrt(128)):
        super().__init__()
        self.query = torch.nn.Conv2d(query_channels, n_head * key_channels,
                                     kernel_size=1, stride=1, padding=0, bias=False)
        self.key = torch.nn.Conv2d(key_channels, n_head * key_channels,
                                   kernel_size=1, stride=1, padding=0, bias=False)
        self.value = torch.nn.Conv2d(value_channels, n_head * value_channels,
                                     kernel_size=1, stride=1, padding=0, bias=False)
        self.dropout = torch.nn.Dropout(0.5)
 
    def forward(self, querys, keys, values):
        b, qc, qh, qw = querys.size()
        _, kc, kh, kw = keys.size()
        _, _, vh, vw = values.size()
 
        bs = 1
        k_channels = kc // bs
        q_channels = qc // bs
        kv_channels = kc // bs
        v_channels = v_channels // bs
        n_head = self.n_head
        key = self.key(keys)
        query = self.query(querys)
        value = self.value(values)

        sliced_key = key.permute(0, 2, 3, 1)
        sliced_key = sliced_key.contiguous().view(-1, k_channels)
        sliced_query = query.permute(0, 2, 3, 1)
        sliced_query = sliced_query.view(-1, query_channels)
        sliced_value = value.permute(0, 2, 3, 1)
        sliced_value = sliced_value.view(-1, value_channels)
        unstacked_result = torch.bmm(sliced_query, self.scaled_factor * sliced_key)
        unstacked_result = torch.nn.functional.softmax(unstacked_result, dim=-1)
        result = self.apply_dropout(unstacked_result, dropout_p=0.5)
        result = torch.bmm(sliced_value, result)

        self.unstacked_result = unstacked_result

        ret = result.view(bs, n_head, vh * vw, v_channels)
        ret = ret.permute(0, 1, 3, 2)
        ret = ret.contiguous().view(bs, n_head * v_channels, vh, vw)
        return ret
