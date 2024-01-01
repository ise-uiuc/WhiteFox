import json
from pathlib import Path
from feedback_loop import update_example_pool, sample_example_pool, gen_prompt
from pprint import pprint

def test_update_example_pool():
    json_file = Path("example-debug.json")
    llvm_source = Path("")

    with open(json_file, "r") as file:
        data = json.load(file)

    example_pool = {}
    # Change the path in data.
    for k in data.keys():
        for idx, _ in enumerate(data[k]["hints"]):
            data[k]["hints"][idx]["codes"][0] = (
                llvm_source / data[k]["hints"][idx]["codes"][0]
            )
            try:
                data[k]["hints"][idx]["codes"][1] = (
                    llvm_source / data[k]["hints"][idx]["codes"][1]
                )
            except:
                pass
            data[k]["hints"][idx]["examples"][0] = (
                llvm_source / data[k]["hints"][idx]["examples"][0]
            )
            data[k]["hints"][idx]["specific_ir"] = (
                llvm_source / data[k]["hints"][idx]["specific_ir"]
            )
            example_pool[f"{k}_oneshot_{idx}"] = {}
    
    # Load the statistics.
    with open("/JawTitan/whitefox-data/starcoder-rl/llvm-opt-1003-debug-2/step2_trigger/statistics-step2.json", "r") as file:
        statistics = json.load(file)
    
    update_example_pool(example_pool, statistics, chosen={})
    pprint(example_pool)
    
    chosen, examples_dict = sample_example_pool(example_pool, 3)

    # Generate the prompt for the first time.
    gen_prompt(data, Path("test-prompt"), examples_dict)

    

if __name__ == "__main__":
    test_update_example_pool()