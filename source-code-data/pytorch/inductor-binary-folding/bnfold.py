_binary_ops = [aten.add.Tensor, aten.sub.Tensor, aten.mul.Tensor, aten.div.Tensor]
_computation_calls = [CallFunction(aten.convolution.default, *_conv_args, _users=1)]

CallFunction(binary_op, _computation_call, KeywordArg("other"))