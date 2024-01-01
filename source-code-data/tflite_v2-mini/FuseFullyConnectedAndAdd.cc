// This function removes explicit broadcasting on type1 and returns whether if
// the reduced `type1` dimensions are the same as the ending dimensions
// of `type2`.
bool IsReducedTailOfShape(Type type1, Type type2) {
  auto tail_type = type1.dyn_cast<ShapedType>();
  auto full_type = type2.dyn_cast<ShapedType>();
  if (!tail_type || !full_type || !tail_type.hasRank() || !full_type.hasRank())
    return false;

  auto i1 = tail_type.getShape().rbegin();
  auto reduced_e1 = tail_type.getShape().rend();
  auto i2 = full_type.getShape().rbegin();

  while ((std::distance(i1, reduced_e1) > 0) && (*(reduced_e1 - 1) == 1)) {
    reduced_e1--;
  }

  return (std::distance(i1, reduced_e1) > 0) &&
         (std::distance(i1, reduced_e1) <= full_type.getRank()) &&
         (std::equal(i1, reduced_e1, i2));
}

bool IsLastDimEqualToNumElements(Type type1, Type type2) {
  return (type1.cast<ShapedType>().getRank() >= 1 &&
          type1.cast<ShapedType>().getDimSize(
              type1.cast<ShapedType>().getRank() - 1) ==
              type2.cast<ShapedType>().getNumElements());
}

// Fuse Add with proceeding FullyConnected.
struct FuseFullyConnectedAndAdd : public OpRewritePattern<TFL::AddOp> {
  using OpRewritePattern<TFL::AddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::AddOp add_op,
                                PatternRewriter &rewriter) const override {
    // Match Add.
    DenseElementsAttr added_value;
    Value constant_val = add_op.getRhs();
    if (!matchPattern(constant_val, m_Constant(&added_value))) return failure();

    // Match Fully Connected.
    auto fc_op = dyn_cast_or_null<TFL::FullyConnectedOp>(
        add_op.getLhs().getDefiningOp());
    if (!fc_op) return failure();

    // Check if the constant RHS is either 0D (scalar), or a 1D with
    // `{num_channels}` shape.
    auto constant_val_type = constant_val.getType().cast<TensorType>();

    // In TFLite FullyConnect definition, bias must be a 1D tensor where
    // the number of elements is equal to the number of channels.
    // If it's not 1D or 0D (which can be broadcasted to 1D), reject the
    // matching.
    bool is_scalar_rhs = false;
    if (constant_val_type.getRank() == 0) {
      is_scalar_rhs = true;
    }

    Value filter = fc_op.getFilter();
    Value bias = fc_op.getBias();
    ElementsAttr bias_value;
    const bool is_none_bias = bias.getType().isa<NoneType>();
    if (fc_op.getFusedActivationFunction() != "NONE") return failure();

    if (!is_none_bias && !matchPattern(bias, m_Constant(&bias_value)))
      return failure();

    // Rewrite
    if (is_none_bias) {
      if (constant_val_type.getRank() == 1) {
        // If there no pre-existing bias and the `constant_val` is 1D, simply
        // use `constant_val` as bias.
        bias = constant_val;
      } else {
        if (!is_scalar_rhs &&
            !(IsReducedTailOfShape(constant_val.getType(), filter.getType()) &&
              IsLastDimEqualToNumElements(filter.getType(),
                                          constant_val.getType()))) {
          return failure();
        }

        // If the `constant_val` is scalar, we must the shape of filter
        // to properly broadcast the scalar to `{num_channels}` shape.

        // Get the number of channels if possible.
        auto filter_type = filter.getType().dyn_cast<RankedTensorType>();
        // Filter must be a `2D` tensor with `{num_channels, num_features}`
        // shape. The following check is rejecting unknown rank (-1).
        if (filter_type == nullptr || filter_type.getRank() != 2) {
          return failure();
        }
      }
    }

    return success();
  }
};