# input_file = "../datasets_files/wmt16-de-en.txt"

# i = 0
# j = 0
# with open(input_file, "r") as f:
#     for line in f.readlines():
#         if i == 10:
#             break
#         with open(f"wmt16-de-en/{i}.txt", "a") as ff:
#             ff.write(line)
#             j += 1
#             if j == 10:
#                 j = 0
#                 i += 1
            
# import json

# input_file = "../datasets_files/cnn_dailymail.json"

# i = 0
# with open(input_file, "r") as f:
#     data_arr = json.load(f)
#     output = []
#     for prompt in data_arr:
#         if i == 10:
#             break
        
#         output.append(prompt)
        
#         if len(output) == 10:
#             with open(f"cnn_dailymail/{i}.json", "w") as ff:
#                 json.dump(output, ff)
#                 output = []
#                 i += 1
            

out = []
for i in range(0, 10):
    for config in [{"name": "piqa", "type": "QA", "fileT": "txt"}, {"name": "wmt16-de-en", "type": "translate_de_en", "fileT": "txt"}, {"name": "cnn_dailymail", "type": "summarize", "fileT": "json"}]:
        obj = {
            "output_file": f"{config['name']}/{i}.statistics",
            "dataset_file": f"{config['name']}/{i}.{config['fileT']}",
            "task_type": config['type']
        }    
        out.append(obj)    
        
        
print(out)