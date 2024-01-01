// Reshape(A, S->S) ==> A
StatusOr<bool> IdentityReshapeRemoving(HloInstruction* reshape) {
  if (reshape->opcode() != HloOpcode::kReshape) {
    return false;
  }
  auto operand = reshape->mutable_operand(0);
  if (Shape::Equal()(reshape->shape(), operand->shape())) {
    TF_RETURN_IF_ERROR(reshape->ReplaceAllUsesWith(operand));
    return true;
  }
  return false;
}