// Returns whether the given type `a` is broadcast-compatible with `b`.
bool IsBroadcastableElementsAttrAndType(Type a, Type b) {
  return OpTrait::util::getBroadcastedType(a, b) != Type();
}

// Fuse Mul with proceeding FullyConnected.
// TODO(b/136285429): Move to tablegen when variadic is supported
struct FuseFullyConnectedAndMul : public OpRewritePattern<TFL::MulOp> {
  using OpRewritePattern<TFL::MulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::MulOp mul_op,
                                PatternRewriter &rewriter) const override {
    
    // If we are broadcasting on the lhs then don't fold the multiply as it
    // would increase the amount of compute done by the fully connected op.
    if (mul_op.getLhs().getType() != mul_op.getType()) return failure();

    // Mul.
    DenseElementsAttr cst;
    Value constant_val = mul_op.getRhs();
    if (!matchPattern(constant_val, m_Constant(&cst))) return failure();

    // Fully Connected.
    auto fc_op = dyn_cast_or_null<TFL::FullyConnectedOp>(
        mul_op.getLhs().getDefiningOp());
    if (!fc_op) return failure();
    Value filter = fc_op.getFilter();
    Value bias = fc_op.getBias();
    ElementsAttr cst_tmp;
    if (!matchPattern(filter, m_Constant(&cst_tmp))) return failure();
    if (!bias.getType().isa<NoneType>() &&
        !matchPattern(bias, m_Constant(&cst_tmp)))
      return failure();
    if (fc_op.getFusedActivationFunction() != "NONE") return failure();

    // Only fuse multiplier if all dimensions other than the depth dimension
    // are equal to 1 since otherwise
    // `matmul(x, filter) * cst != matmul(x, filter * cst)`
    // even if `filter` and `cst` are be broadcastable.
    auto shape = cst.getType().getShape();
    if (!IsDimensionsDegenerateExceptLastOne(shape)) return failure();

    int64_t element_size = shape.empty() ? 1 : shape[shape.size() - 1];
    // Expand and transpose the multiplier since weights are using the
    // OHWI data format in TFLite.
    int64_t normalized_shape[2] = {element_size, 1};
    auto new_cst = cst.reshape(RankedTensorType::get(
        normalized_shape, cst.getType().getElementType()));
    Type new_type = new_cst.getType();
    if (!IsBroadcastableElementsAttrAndType(new_type, filter.getType())) {
      return failure();
    }
    
    return success();
  }
};
