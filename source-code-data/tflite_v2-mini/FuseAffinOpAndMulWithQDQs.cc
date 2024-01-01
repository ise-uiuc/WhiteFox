template <typename AffineOpType>
struct FuseAffinOpAndMulWithQDQs : public OpRewritePattern<TFL::MulOp> {
  using OpRewritePattern<TFL::MulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::MulOp mul_op,
                                PatternRewriter &rewriter) const override {

    // Mul. Required 1-D rhs for batch normalization.
    DenseElementsAttr gamma_cst;
    Value gamma = mul_op.getRhs();
    if (!matchPattern(gamma, m_Constant(&gamma_cst))) return failure();
    if (gamma_cst.getType().getRank() != 1) return failure();

    // Affine op
    Operation *mul_op_lhs = mul_op.getLhs().getDefiningOp();
    auto fc_op = dyn_cast_or_null<AffineOpType>(mul_op_lhs);
    if (!fc_op) return failure();
    Value filter = fc_op.getFilter();
    Value bias = fc_op.getBias();

    // QDQs
    auto dq_op = dyn_cast_or_null<TFL::DequantizeOp>(filter.getDefiningOp());
    if (!dq_op) return failure();
    auto q_op =
        dyn_cast_or_null<TFL::QuantizeOp>(dq_op.getInput().getDefiningOp());
    if (!q_op) return failure();
    filter = q_op.getInput();

    // weight constant
    ElementsAttr cst_tmp;
    if (!matchPattern(filter, m_Constant(&cst_tmp))) return failure();
    if (!bias.getType().isa<NoneType>() &&
        !matchPattern(bias, m_Constant(&cst_tmp)))
      return failure();
    if (fc_op.getFusedActivationFunction() != "NONE") return failure();

    // Broadcast the constant operand of Mul if it isn't compatible to the
    // filter input. We only support broadcasting the operand along the depth
    // dimension, when the operand's depth is 1.
    rewriter.setInsertionPoint(q_op);
    Location loc = fc_op.getLoc();
    Value broadcasted_gamma;
    if (isa<TFL::Conv2DOp>(mul_op_lhs)) {
      auto mul_rhs = ExpandTo4DForConv(gamma_cst);
      broadcasted_gamma = rewriter.create<ConstOp>(loc, mul_rhs);
    } else if (isa<TFL::DepthwiseConv2DOp>(mul_op_lhs)) {
      auto mul_rhs = ExpandTo4DForDepthwiseConv(gamma_cst);
      broadcasted_gamma = rewriter.create<ConstOp>(loc, mul_rhs);
    } else {
      return failure();
    }

    // Make sure that the fused bias will be a 1D tensor.
    auto gamma_shape = gamma.getType().cast<ShapedType>();
    if (!gamma_shape.hasRank() || gamma_shape.getRank() != 1) {
      return failure();
    }

    // Update the scale in the quantize op.
    auto new_qtype = RescaleQtype(q_op.getQtype(), gamma_cst);
    if (!new_qtype) return failure();
    
    return success();
  }
};