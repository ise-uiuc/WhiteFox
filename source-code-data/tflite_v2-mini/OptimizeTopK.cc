// Reduce the K of a TopKV2Op for the following case.
//
// values, indices = tfl.topkv2(%inputs, K)
// %1 = tfl.slice(values, 0, k)
// %2 = tfl.slice(indices,0, k)
// .... (values and indices only used for %1 and %2)
//
// %1 or %2 can be absent. If values and indices are only used here,
// this pattern can be replaced with (conceptually)
//
// %values, %indices = tfl.topkv2(%inputs, k)
// replace all use of %1 with values
// replace all use of %2 with indices
//
struct OptimizeTopK : public OpRewritePattern<TFL::TopKV2Op> {
  using OpRewritePattern::OpRewritePattern;

  // It computes the last dim k of slice size of value.user.
  // If value has no use then return 0.
  std::optional<int32_t> ComputeSliceK(Value value) const {
    if (value.use_empty()) return 0;
    auto slice_op =
        llvm::dyn_cast_or_null<TFL::SliceOp>(value.getUses().begin().getUser());
    // We only match for the case where value is used by SliceOp.
    if (!slice_op) return std::nullopt;
    DenseElementsAttr begin;
    DenseElementsAttr size;
    if (!matchPattern(slice_op->getOperand(1), m_Constant(&begin)) ||
        !matchPattern(slice_op->getOperand(2), m_Constant(&size)))
      return std::nullopt;

    // Check if "begin" is a zero tensor.
    for (auto begin_idx : begin.getValues<APInt>())
      if (begin_idx != 0) return std::nullopt;

    // Check if "size" is equal to slice_op.input.shape except
    // for last dimension.
    // It can be done  by verifying the number of elements:
    // i.e., num_input/input_last_dim = num_result/k
    auto input_ty = value.getType().dyn_cast_or_null<ShapedType>();
    auto result_ty = slice_op.getType().dyn_cast<ShapedType>();
    if (!input_ty || !result_ty) return std::nullopt;
    if (!input_ty.hasStaticShape() || !result_ty.hasStaticShape())
      return std::nullopt;
    if (!input_ty.getRank() || !result_ty.getRank()) return std::nullopt;
    int num_input = input_ty.getNumElements();
    int input_last_dim = input_ty.getShape().back();
    if (input_last_dim < 1) return std::nullopt;
    int num_result = result_ty.getNumElements();
    auto size_last = *(--size.value_end<APInt>());
    int32_t k = size_last.getSExtValue();
    if (num_input / input_last_dim * k != num_result) return std::nullopt;
    // We don't match sliceOp with last dim size = 0.
    if (!k) return std::nullopt;
    return k;
  }

  LogicalResult matchAndRewrite(TFL::TopKV2Op op,
                                PatternRewriter &rewriter) const override {
    auto values = op.getValues();
    auto indices = op.getIndices();
    // op.getValues() and op.getIndices() cannot be used more than once.
    
    if (!values.hasOneUse() && !values.use_empty()) return failure();
    if (!indices.hasOneUse() && !indices.use_empty()) return failure();

    auto k_values_or = ComputeSliceK(values);
    auto k_indices_or = ComputeSliceK(indices);
    if (!k_values_or.has_value() || !k_indices_or.has_value()) return failure();
    int32_t k_values = k_values_or.value();
    int32_t k_indices = k_indices_or.value();
    // We don't match two SliceOp with different sizes.
    if (k_values != k_indices && !values.use_empty() && !indices.use_empty())
      return failure();

    return success();
  }
};