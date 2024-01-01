BroadcastCanonicalizer::BroadcastCanonicalizer() {}

StatusOr<bool> BroadcastCanonicalizer::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;

  // Sort broadcast dims. Then insert a transpose on the broadcast to get the
  // original shape back.
  for (const auto& computation :
       module->MakeNonfusionComputations(execution_threads)) {
    for (HloInstruction* hlo : computation->MakeInstructionPostOrder()) {
      if (hlo->opcode() != HloOpcode::kBroadcast) {
        continue;
      }
      if (absl::c_is_sorted(hlo->dimensions())) {
        continue;
      }
      std::vector<int64_t> new_dims(hlo->dimensions().begin(),
                                    hlo->dimensions().end());
      std::vector<int64_t> original_dims(hlo->dimensions().begin(),
                                         hlo->dimensions().end());

      std::vector<int64_t> new_broadcast_dims(hlo->shape().dimensions().begin(),
                                              hlo->shape().dimensions().end());
      absl::c_sort(new_dims);
      const int64_t rank = hlo->shape().rank();
      for (int i = 0; i < new_dims.size(); ++i) {
        new_broadcast_dims[new_dims[i]] =
            hlo->operand(0)->shape().dimensions(i);
      }

      auto new_broadcast = MakeBroadcastHlo(hlo->mutable_operand(0), new_dims,
                                            new_broadcast_dims);
      std::vector<int64_t> transpose_dims(rank);
      absl::c_iota(transpose_dims, 0);
      for (int i = 0; i < new_dims.size(); ++i) {
        transpose_dims[new_dims[i]] = new_dims[std::distance(
            original_dims.begin(), absl::c_find(original_dims, new_dims[i]))];
      }
      TF_ASSIGN_OR_RETURN(new_broadcast,
                          MakeTransposeHlo(new_broadcast, transpose_dims));
      TF_RETURN_IF_ERROR(computation->ReplaceInstruction(hlo, new_broadcast));
      changed = true;
    }
  }
  return changed;
}
