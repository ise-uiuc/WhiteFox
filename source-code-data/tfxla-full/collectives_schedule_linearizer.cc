namespace xla {

// TODO(b/181653482): Fix for interprocedural collectives as well.
StatusOr<bool> CollectivesScheduleLinearizer::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  if (is_enabled_ && !is_enabled_(module)) {
    return false;
  }
  bool changed = false;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    std::unique_ptr<HloReachabilityMap> reachability;
    HloInstruction* prev_done = nullptr;
    for (HloInstruction* inst : computation->MakeInstructionPostOrder()) {
      auto* next = DynCast<HloCollectiveInstruction>(inst);
      if (!next) {
        continue;
      }
      // Build reachability map on demand if we actually see collectives.
      if (!reachability) {
        reachability = HloReachabilityMap::Build(computation);
      }
      // Derive the 'start' and 'done' peers of this instruction. For non-async
      // variants of collectives, they are the same as this instruction. For
      // async variants, the start is this instruction and the 'done' is the
      // matching async-done instruction.
      HloInstruction* start = next;
      HloInstruction* done = next;
      switch (next->opcode()) {
        case HloOpcode::kAllReduceStart:
        case HloOpcode::kAllGatherStart:
        case HloOpcode::kCollectivePermuteStart:
        case HloOpcode::kAsyncStart:
          // Find the async-done corresponding to this async start instruction.
          CHECK_EQ(start->user_count(), 1);
          done = start->users()[0];
          break;
        default:
          break;
      }

      if (prev_done && !reachability->IsConnected(start, prev_done)) {
        // If prev_done and start are independent, enforce ordering.
        TF_RETURN_IF_ERROR(prev_done->AddControlDependencyTo(next));
        VLOG(1) << "Adding control dependency from " << prev_done->ToString()
                << " to " << start->ToString();
        changed = true;
      }
      prev_done = done;
    }
  }
  return changed;
}

}  // namespace xla
