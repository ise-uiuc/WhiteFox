// If the operand to a broadcastable op is a splat constant, try to replace it
// with a 0-d constant, e.g. before this optimization,
//   %cst = arith.constant dense<1.0> : tensor<16x16x4xf32>
//   %0 = "tfl.conv_2d"...
//   %1 = "tfl.add"(%0, %cst) : (tensor<16x16x4xf32>, tensor<16x16x4xf32>)
// After this optimization:
//   %cst = arith.constant dense<1.0> : tensor<f32>
//   %0 = "tfl.conv_2d"...
//   %1 = "tfl.add"(%0, %cst) : (tensor<16x16x4xf32>, tensor<f32>)
// This pattern can enable more fusing opportunities when the binary op is
// following conv ops.
template <typename BinaryOpType>
struct ScalarizeSplatConstantForBroadcastableOps
    : public OpRewritePattern<BinaryOpType> {
  using OpRewritePattern<BinaryOpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(BinaryOpType binary_op,
                                PatternRewriter &rewriter) const override {

    DenseElementsAttr splat_elements_attr;
    if (!IsScalarizableSplatConstant(binary_op.getRhs(),
                                     &splat_elements_attr)) {
      return failure();
    }

    constexpr int kSplatOperandIndex = 1;
    auto result_type =
        binary_op.getResult().getType().template cast<ShapedType>();
    mlir::Value non_splat_operand =
        binary_op.getOperand(1 - kSplatOperandIndex);
    auto non_splat_operand_type =
        non_splat_operand.getType().cast<ShapedType>();
    // If the other operand's shape does not equal to the result shape, then we
    // cannot scalarize the splat constant because the result shape relies on
    // the splat constant op's shape for broadcasting.
    if (!non_splat_operand_type.hasStaticShape() ||
        non_splat_operand_type.getShape() != result_type.getShape() ||
        non_splat_operand_type.getRank() > 4) {
      return failure();
    }

    // If non-splat operand is not fusable affine ops, then no need to apply
    // this transformation.
    if (!CanFuseAffineOp(non_splat_operand.getDefiningOp(), binary_op)) {
      return failure();
    }

    // Creates a new scalar constant op using the splat value.
    mlir::Value splat_operand = binary_op.getOperand(kSplatOperandIndex);
    auto scalar_elements_attr = DenseElementsAttr::get(
        RankedTensorType::get({},
                              splat_elements_attr.getType().getElementType()),
        splat_elements_attr.getSplatValue<mlir::Attribute>());

    auto scalar_constant_op = rewriter.create<arith::ConstantOp>(
        splat_operand.getLoc(), scalar_elements_attr.getType(),
        scalar_elements_attr);

    binary_op.setOperand(kSplatOperandIndex, scalar_constant_op);

    return success();
  }

 private:
  // Returns true if this value is a splat constant op which can be scalarized.
  // Also returns the elements attr if this value is indeed a splat constant.
  bool IsScalarizableSplatConstant(mlir::Value value,
                                   DenseElementsAttr *elements_attr) const {
    if (!matchPattern(value, m_Constant(elements_attr))) {
      return false;
    }
    auto element_type = value.getType().cast<ShapedType>().getElementType();
    // Ignore per-axis quantized constants because after converting to scalar,
    // we will lose per-axis qantization parameter.
    if (element_type.isa<quant::UniformQuantizedPerAxisType>()) {
      return false;
    }
    if (IsScalar(value)) {
      return false;
    }
    return elements_attr->isSplat();
  }

  // If this type is a scalar shaped type.
  bool IsScalar(mlir::Value value) const {
    auto type = value.getType().dyn_cast<ShapedType>();
    if (!type) {
      return false;
    }
    if (!type.hasStaticShape()) {
      return false;
    }
    return type.getNumElements() == 1;
  }

  // Returns true if we can fuse an affine op with consuming binary op.
  bool CanFuseAffineOp(Operation *affine_op, Operation *binary_op) const {
    if (!isa_and_nonnull<TFL::Conv2DOp, TFL::DepthwiseConv2DOp,
                         TFL::FullyConnectedOp>(affine_op)) {
      return false;
    }
    DenseElementsAttr value;
    // Check that bias are constants if not none.
    Value bias = affine_op->getOperand(2);
    if (!bias.getType().isa<NoneType>() &&
        !matchPattern(bias, m_Constant(&value))) {
      return false;
    }
    // If the binary op is mul/div, also check that filter is constant.
    if (isa<TFL::MulOp, TFL::DivOp>(binary_op) &&
        !matchPattern(affine_op->getOperand(1), m_Constant(&value))) {
      return false;
    }

    // We can only fuse F32/BF16.
    auto is_fusable_type = [](Type t) {
      Type element_type = t;
      if (auto shaped_type = t.dyn_cast<ShapedType>()) {
        element_type = shaped_type.getElementType();
      }
      return element_type.isBF16() || element_type.isF32();
    };
    for (Type t : binary_op->getOperandTypes()) {
      if (!is_fusable_type(t)) {
        return false;
      }
    }

    return true;
  }
};