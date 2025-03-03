#!/usr/bin/env python3
import json
import os
import re

# Directory containing function body files
func_body_dir = "source-code-data/llvm/llvm-func-body"

# Target lines for specific passes (add as many as you know)
target_lines = {
    "ADCEPass": "Worklist.push_back(&I);",
    "ArgumentPromotionPass": "C.getOuterRefSCC().replaceNodeFunction(N, *NewF);",
    "ConstantMergePass": "GV.eraseFromParent();",
    "DCEPass": "I->eraseFromParent();",
    "DeadArgumentEliminationPass": "F.replaceAllUsesWith(ConstantExpr::getBitCast(NF, F.getType()));",
    "DSEPass": "State.deleteDeadInstruction(DeadI);",
    "GlobalDCEPass": "F->replaceNonMetadataUsesWith(ConstantPointerNull::get(F->getType()));",
    "GlobalOptPass": "ChangeCalleesToFastCall(&F);",
    "LoopFusePass": "Preheader = InsertPreheaderForLoop(L, DT, LI, MSSAU, PreserveLCSSA);",
    "GVNPass": "Changed |= performScalarPRE(CurInst);",
    "LCSSAPass": "Builder.SetInsertPoint(&ExitBB->front());",
    "LICMPass": "U = PoisonValue::get(I.getType());",
    "LoopDeletionPass": "breakLoopBackedge(L, DT, SE, LI, MSSA);",
    "LoopExtractorPass": "LI.erase(L);",
    "LoopSimplifyCFGPass": "bool Changed = BranchFolder.run();",
    "SimpleLoopUnswitchLegacyPass": "UnswitchCB(/*CurrentLoopValid*/ true, false, {});",
    "AggressiveInstCombinePass": "ReduceExpressionGraph(NewDstSclTy);",
    "EarlyCSEPass": "Inst.replaceAllUsesWith(V);",
    "InstSimplifyPass": "DeadInstsInBB.push_back(&I);",
    "PromotePass": "PromoteMemToReg(Allocas, DT, &AC);",
    "TestPass": "int result = x + y;"
}

# Create JSON structure
json_data = {}

# Get all pass names from function body files
for file in os.listdir(func_body_dir):
    if file.endswith('.cpp'):
        # Extract pass name and remove trailing digits
        base_name = os.path.splitext(file)[0]
        pass_name = re.sub(r'\d+$', '', base_name)
        
        # Skip empty pass name
        if not pass_name:
            continue
        
        # Use known target line if available, otherwise use a default
        target_line = target_lines.get(pass_name, "// No specific target line provided")
        
        # Add to JSON structure
        json_data[pass_name] = {
            "hints": [
                {
                    "type": "trigger",
                    "target_line": target_line
                }
            ]
        }

# Write JSON to file
with open("llvm_target_fixed.json", "w") as f:
    json.dump(json_data, f, indent=2)

print(f"Created JSON with {len(json_data)} pass entries")