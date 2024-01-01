// Reshape(Broadcast(A, []->[1]), [1]->[]) ==> A
StatusOr<bool> ReshapeBroadcastForwarding(HloInstruction* reshape) {
  if (reshape->opcode() != HloOpcode::kReshape) {
    return false;
  }
  auto broadcast = reshape->mutable_operand(0);
  if (broadcast->opcode() != HloOpcode::kBroadcast) {
    return false;
  }

  if (reshape->shape().rank() != 0) {
    return false;
  }

  if (broadcast->shape().rank() != 1) {
    return false;
  }

  if (broadcast->mutable_operand(0)->shape().rank() != 0) {
    return false;
  }

  TF_RETURN_IF_ERROR(
      reshape->ReplaceAllUsesWith(broadcast->mutable_operand(0)));

  return true;
}