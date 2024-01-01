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
// TODO(b/136285429): Move to tablegen when variadic is supported
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
        int num_channels = filter_type.getShape()[0];

        // Create a zero tensor with shape {num_channels}, and the type need
        // to be the same as constant_val. This is a way to gracefully handle
        // scalar tensor. The Add will always be constant-folded away
        // regardless if `constant_val` is a scalar or not.
        RankedTensorType type = RankedTensorType::get(
            {num_channels}, constant_val_type.getElementType());
        auto attr = rewriter.getZeroAttr(type);
        bias = rewriter.create<arith::ConstantOp>(add_op.getLoc(), type, attr);
        auto none_af = rewriter.getStringAttr("NONE");
        if (is_scalar_rhs) {
          bias =
              rewriter
                  .create<AddOp>(add_op.getLoc(), bias, constant_val, none_af)
                  .getOutput();
        } else {
          // If the RHS is neither a scalar constant nor a 1d constant, look
          // if there is opportunity to reduce the dimentionality and allow
          // implicit broadcasting

          auto new_added_value = added_value.reshape(RankedTensorType::get(
              {added_value.getType().cast<ShapedType>().getNumElements()},
              added_value.getType().cast<ShapedType>().getElementType()));

          ::mlir::arith::ConstantOp new_constant_val =
              rewriter.create<::mlir::arith::ConstantOp>(
                  add_op.getLoc(),
                  /*value=*/new_added_value);

          bias = rewriter
                     .create<::mlir::TFL::AddOp>(
                         add_op.getLoc(),
                         /*lhs=*/bias,
                         /*rhs=*/new_constant_val.getResult(),
                         /*fused_activation_function=*/none_af)
                     .getOutput();
        }
      }
    } else {
      bias = rewriter
                 .create<AddOp>(add_op.getLoc(), bias, constant_val,
                                rewriter.getStringAttr("NONE"))
                 .getOutput();
    }

    auto fc = rewriter.create<TFL::FullyConnectedOp>(
        FusedLoc::get(fc_op.getContext(), {fc_op.getLoc(), add_op.getLoc()}),
        add_op.getType(),
        /*input=*/fc_op.getInput(),
        /*filter=*/filter,
        /*bias=*/bias,
        /*fused_activation_function=*/
        rewriter.getStringAttr(add_op.getFusedActivationFunction()),
        /*weights_format=*/rewriter.getStringAttr(fc_op.getWeightsFormat()),
        /*keep_num_dims=*/rewriter.getBoolAttr(fc_op.getKeepNumDims()),
        /*asymmetric_quantize_inputs=*/
        fc_op.getAsymmetricQuantizeInputsAttr());
    rewriter.replaceOp(add_op, fc.getOutput());

    return success();
  }
};