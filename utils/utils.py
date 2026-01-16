import json 

def read_jsonl(file_path):
    """Reads a JSONL file and returns a list of dictionaries."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def write_jsonl(file_path, rows):
    """Writes a list of dictionaries to a JSONL file."""
    with open(file_path, 'w') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            
def extract_hash_answer(text: str):
    if "####" not in text:
        return None
    return text.split("####")[1].strip()