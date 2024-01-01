def _combined_fusion(computation_call, elementwise_op):
    return CallFunction(elementwise_op, computation_call)