namespace {

// Replaces all uses of old_instr with new_instr except the use at
// `while_body_root` (which must be a tuple instruction) at index `tuple_index`.
// This utility helps us replace an instruction in the while body with a
// constant while still keeping it trivially loop invariant.
Status ReplaceUsesWhileKeepingLoopInvariance(HloInstruction* old_instr,
                                             HloInstruction* new_instr,
                                             HloInstruction* while_body_root,
                                             int64_t tuple_index) {
  CHECK_EQ(while_body_root->opcode(), HloOpcode::kTuple);

  std::vector<HloInstruction*> users;
  users.reserve(old_instr->user_count());
  absl::c_copy(old_instr->users(), std::back_inserter(users));

  for (auto* user : users) {
    for (int64_t i = 0, e = user->operand_count(); i < e; i++) {
      if (user->operand(i) == old_instr &&
          !(user == while_body_root && i == tuple_index)) {
        TF_RETURN_IF_ERROR(user->ReplaceOperandWith(i, new_instr));
      }
    }
  }

  return OkStatus();
}

HloInstruction* CloneHelper(const HloInstruction* instruction,
                            HloComputation* computation) {
  if (instruction->opcode() == HloOpcode::kConstant) {
    return computation->AddInstruction(instruction->Clone(/*suffix=*/".sunk"));
  }
  if (instruction->opcode() == HloOpcode::kBroadcast) {
    return computation->AddInstruction(instruction->CloneWithNewOperands(
        instruction->shape(),
        {CloneHelper(instruction->operand(0), computation)}));
  }
  LOG(FATAL) << "Unexpected instruction.";
}

}  // namespace

StatusOr<bool> WhileLoopConstantSinking::TrySinkingConstantsIntoWhileLoop(
    HloInstruction* while_instr) {
  HloComputation* while_cond = while_instr->while_condition();
  HloComputation* while_body = while_instr->while_body();

  const HloInstruction& init_value = *while_instr->operand(0);
  if (init_value.opcode() != HloOpcode::kTuple) {
    return false;
  }

  bool changed = false;

  absl::flat_hash_map<int64_t, absl::InlinedVector<HloInstruction*, 1>>
      conditional_gte_index_to_insts =
          WhileUtil::GetGTEsMapForWhileConditional(*while_cond);
  std::vector<HloInstruction*> invariant_body_gtes =
      WhileUtil::GetInvariantGTEsForWhileBody(*while_body);

  for (HloInstruction* invariant_body_gte : invariant_body_gtes) {
    int64_t index = invariant_body_gte->tuple_index();
    const HloInstruction& invariant_value = *init_value.operand(index);

    // Original value should be a constant or broadcast of constant.
    if (invariant_value.opcode() != HloOpcode::kConstant &&
        (!sink_broadcast_of_constants_ ||
         invariant_value.opcode() != HloOpcode::kBroadcast ||
         invariant_value.operand(0)->opcode() != HloOpcode::kConstant)) {
      continue;
    }

    // Sink into the while_body.
    // Should have at least one user that's not while_body_root.
    if (invariant_body_gte->user_count() > 1) {
      HloInstruction* constant_instr =
          CloneHelper(&invariant_value, while_body);
      TF_RETURN_IF_ERROR(ReplaceUsesWhileKeepingLoopInvariance(
          invariant_body_gte, constant_instr, while_body->root_instruction(),
          index));
      changed = true;
    }

    // Check if there is a corresponding GTE in while_conditional.
    auto it = conditional_gte_index_to_insts.find(index);
    if (it == conditional_gte_index_to_insts.end()) {
      continue;
    }

    for (HloInstruction* invariant_cond_gte : it->second) {
      // Should have at least one user.
      if (invariant_cond_gte->user_count() > 0) {
        HloInstruction* constant_instr =
            CloneHelper(&invariant_value, while_cond);
        TF_RETURN_IF_ERROR(
            invariant_cond_gte->ReplaceAllUsesWith(constant_instr));
        changed = true;
      }
    }
  }

  return changed;
}

