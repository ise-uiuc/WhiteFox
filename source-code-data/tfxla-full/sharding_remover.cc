namespace xla {

// Remove Sharding custom-call instruction by assigning its users to
// to its operand.
StatusOr<bool> ShardingRemover::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;

  const absl::flat_hash_set<absl::string_view> to_remove_sharding_ops = {
      "Sharding", "SPMDShardToFullShape", "SPMDFullToShardShape"};

  for (HloComputation* computation : module->computations(execution_threads)) {
    auto instructions = computation->MakeInstructionPostOrder();
    std::reverse(instructions.begin(), instructions.end());
    for (HloInstruction* instruction : instructions) {
      if (instruction->opcode() != HloOpcode::kCustomCall) {
        continue;
      }

      if (!to_remove_sharding_ops.contains(instruction->custom_call_target())) {
        continue;
      }
      CHECK(instruction->operand_count() == 1)
          << "Sharding instruction must have exactly one operand";
      TF_RETURN_IF_ERROR(
          instruction->ReplaceAllUsesWith(instruction->mutable_operand(0)));
      changed = true;
    }
  }

  return changed;
}

}  // namespace xla
