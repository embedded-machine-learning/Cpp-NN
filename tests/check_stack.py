import re

def extract_locations_and_memory(filename):
    pattern = re.compile(r"^(.*?:\d+):.*?\b(\d+)\s+[a-zA-Z]+$")
    locations_and_memory = []
    try:
        with open(filename, 'r') as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                match = pattern.match(line)
                if match:
                    location, bytes_used = match.groups()
                    locations_and_memory.append((location, int(bytes_used)))
    except FileNotFoundError:
        print(f"Error: Could not find the file named '{filename}'.")

    sorted_locations_and_memory = sorted(locations_and_memory, key=lambda x: x[1], reverse=True)
    for location, bytes_used in sorted_locations_and_memory:
        print(f"{location:<50} | {bytes_used} bytes")

print("=== BasicAttention ===")
extract_locations_and_memory("BasicAttention.su")
print("=== LARFormer ===")
extract_locations_and_memory("LARFormer.su")
print("=== LinearAttention ===")
extract_locations_and_memory("LinearAttention.su")
print("=== LowRank ===")
extract_locations_and_memory("LowRank.su")
