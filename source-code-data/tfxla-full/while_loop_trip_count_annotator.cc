namespace xla {

StatusOr<bool> WhileLoopTripCountAnnotator::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (const HloComputation* comp : module->computations(execution_threads)) {
    for (HloInstruction* instr : comp->instructions()) {
      if (instr->opcode() != HloOpcode::kWhile) {
        continue;
      }
      if (auto trip_count = ComputeWhileLoopTripCount(instr)) {
        WhileLoopBackendConfig config;
        config.mutable_known_trip_count()->set_n(*trip_count);
        TF_RETURN_IF_ERROR(instr->set_backend_config(config));
        changed = true;
      }
    }
  }
  return changed;
}

}  // namespace xla
