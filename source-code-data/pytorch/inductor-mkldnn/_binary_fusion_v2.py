def _binary_fusion_v2(computation_call, binary_fn):
    return CallFunction(binary_fn, computation_call, KeywordArg("other"))