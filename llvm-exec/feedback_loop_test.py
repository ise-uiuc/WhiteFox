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
    
    # Load mock statistics for testing
    mock_statistics = {
        "all_files": ["test1.c", "test2.c"],  
        "grammatically correct": ["test1.c"],
        "grammatically uncorrect": ["test2.c"],
        "target_lines_triggered": {
            "line1": ["test1.ll"],
            "line2": ["test2.ll"]
        }
    }
    
    # Add mock pass names and their target lines
    for k in data.keys():
        for idx, _ in enumerate(data[k]["hints"]):
            passname = f"{k}_oneshot_{idx}"
            mock_statistics[passname] = f"line{idx+1}"
    
    # Test update_example_pool
    update_example_pool(example_pool, mock_statistics, chosen={})
    print("\nExample pool after update:")
    pprint(example_pool)
    
    # Test sample_example_pool
    chosen, examples_dict = sample_example_pool(example_pool, 3)
    print("\nChosen examples:")
    pprint(chosen)
    print("\nExamples dict:")
    pprint(examples_dict)

    # Test gen_prompt
    print("\nGenerating prompts...")
    gen_prompt(data, Path("test-prompt"), examples_dict)

def test_with_real_data():
    # Test with actual data from statistics file
    json_file = Path("example-debug.json")
    with open(json_file, "r") as file:
        data = json.load(file)

    example_pool = {}
    for k in data.keys():
        for idx, _ in enumerate(data[k]["hints"]):
            example_pool[f"{k}_oneshot_{idx}"] = {}
    
    # Load the actual statistics
    stats_path = Path("ollama/statistics-step2.json")  # Updated path for Ollama
    if stats_path.exists():
        with open(stats_path, "r") as file:
            statistics = json.load(file)
        
        update_example_pool(example_pool, statistics, chosen={})
        print("\nExample pool with real data:")
        pprint(example_pool)
        
        chosen, examples_dict = sample_example_pool(example_pool, 3)
        gen_prompt(data, Path("test-prompt-real"), examples_dict)

if __name__ == "__main__":
    print("Running mock data test...")
    test_update_example_pool()
    print("\nRunning real data test...")
    test_with_real_data()