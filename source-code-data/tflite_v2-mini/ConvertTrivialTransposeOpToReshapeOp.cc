struct ConvertTrivialTransposeOpToReshapeOp
    : public OpRewritePattern<TFL::TransposeOp> {
  using OpRewritePattern<TFL::TransposeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::TransposeOp transpose_op,
                                PatternRewriter &rewriter) const override {


    auto input_type = transpose_op.getInput().getType().cast<ShapedType>();
    auto output_type = transpose_op.getOutput().getType().cast<ShapedType>();
    // It's possible to know if the transformation is safe only if the input
    // & output shapes are fully known and permutation is a constant.
    if (!input_type.hasStaticShape() || !output_type.hasStaticShape())
      return failure();
    Value perm = transpose_op.getPerm();
    DenseElementsAttr perm_values_attr;
    if (!matchPattern(perm, m_Constant(&perm_values_attr))) return failure();

    auto input_shape = input_type.getShape();
    SmallVector<int64_t, 8> perm_values;
    for (const auto &dim : perm_values_attr.getValues<APInt>())
      perm_values.push_back(dim.getSExtValue());

    // This should never happen unless the input graph is malformed.
    if (input_shape.size() != perm_values.size()) {
      transpose_op.emitError(
          "TransposeOP has inconsistent input and perm values.");
    }

    SmallVector<int, 8> old_major_index_ordering;
    SmallVector<int, 8> new_major_index_ordering;
    for (int i = 0, end = input_shape.size(); i < end; i++) {
      if (input_shape[i] != 1) {
        old_major_index_ordering.push_back(i);
      }

      if (input_shape[perm_values[i]] != 1) {
        new_major_index_ordering.push_back(perm_values[i]);
      }
    }
    if (old_major_index_ordering != new_major_index_ordering) {
      return failure();
    }

    return success();
  }
};