import argparse
import requests
import time
import os
import json
from pathlib import Path

def process_msg(msg):
   if "```" not in msg:
       return msg
   code_st = False 
   code = ""
   for line in msg.splitlines():
       if code_st:
           if line.strip().startswith("```"):
               code_st = False
               continue
           code += line + "\n"
       else:
           if line.strip().startswith("```"):
               code_st = True
   return code

def generate_with_ollama(prompt, model="starcoder", temperature=1.0):
   try:
       response = requests.post('http://localhost:11434/api/generate', 
           json={
               "model": model,
               "prompt": prompt,
               "temperature": temperature,
           })
       return response.json()['response']
   except Exception as e:
       print(f"Error calling Ollama API: {e}")
       return None

if __name__ == "__main__":
   parser = argparse.ArgumentParser()
   parser.add_argument("--prompt-dir", type=str, default="prompt/demo")
   parser.add_argument("--outdir", type=str, default="ollama/zero-shot")
   parser.add_argument("--iter", type=int, default=1)
   parser.add_argument("--temperature", type=float, default=1.0)
   parser.add_argument("--batch-size", type=int, default=100)
   parser.add_argument("--prompt-only", action="store_true")
   parser.add_argument("--target", type=str, default="PyTorch")
   parser.add_argument("--model", type=str, default="starcoder")

   args = parser.parse_args()

   prompt_dir = Path(args.prompt_dir)
   opts = {}
   for prompt_file in prompt_dir.iterdir():
       if not prompt_file.is_file():
           continue
       opts[prompt_file.stem] = prompt_file.read_text()

   outdir = args.outdir
   iteration = args.iter
   temperature = args.temperature
   
   for opt_idx, opt in enumerate(opts):
       if os.path.exists(os.path.join(outdir, opt, f"{opt}_1.py")):
           print("Skipping opt ", opt)
           continue

       code_idx = 0
       ret = {"opt": opt}
       ret["response"] = {}
       os.makedirs(os.path.join(outdir, opt), exist_ok=True)
       user_input = opts[opt]

       with open(os.path.join(outdir, opt, f"prompt.txt"), "w") as f:
           f.write(user_input)

       if args.prompt_only:
           print(opt_idx)
           continue

       for i in range(iteration):
           for _ in range(args.batch_size):
               t_start = time.time()
               response = generate_with_ollama(user_input, args.model, temperature)
               if response:
                   g_time = time.time() - t_start
                   print(f"[{opt_idx+1}/{len(opts)}] {opt} used time: ", g_time)
                   
                   code = process_msg(response)
                   code_idx += 1
                   
                   with open(os.path.join(outdir, opt, f"{opt}_{code_idx}.py"), "w") as f:
                       f.write(code)
                   with open(os.path.join(outdir, opt, f"{opt}_{code_idx}.txt"), "w") as f:
                       f.write(response)
                       
                   ret["response"][code_idx] = {
                       "raw": response,
                       "code": code,
                       "g_time": g_time
                   }
               time.sleep(1)  # Rate limiting

       with open(os.path.join(outdir, "outputs.json"), "a") as f:
           f.write(json.dumps(ret, indent=4) + "\n")