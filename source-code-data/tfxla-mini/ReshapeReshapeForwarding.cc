// Reshape(Reshape(A, []->[1]), [1]->[]) ==> A
StatusOr<bool> ReshapeReshapeForwarding(HloInstruction* reshape) {
  if (reshape->opcode() != HloOpcode::kReshape) {
    return false;
  }
  auto reshape_2 = reshape->mutable_operand(0);
  if (reshape_2->opcode() != HloOpcode::kReshape) {
    return false;
  }

  if (!Shape::Equal()(reshape->shape(), reshape_2->operand(0)->shape())) {
    return false;
  }
  TF_RETURN_IF_ERROR(
      reshape->ReplaceAllUsesWith(reshape_2->mutable_operand(0)));
  return true;
}