// Convert(A, T->T) ==> A
StatusOr<bool> IdentityConvertRemoving(HloInstruction* convert) {
  if (convert->opcode() != HloOpcode::kConvert) {
    return false;
  }
  auto operand = convert->mutable_operand(0);
  if (Shape::Equal()(convert->shape(), operand->shape())) {
    TF_RETURN_IF_ERROR(convert->ReplaceAllUsesWith(operand));
    return true;
  }
  return false;
}