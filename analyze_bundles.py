import os
from collections import Counter

bundles_path = r'F:\Dev\RD-Agent-main\RDagentDB\production_bundles'

dirs = [d for d in os.listdir(bundles_path) if os.path.isdir(os.path.join(bundles_path, d))]

print(f"Total directories: {len(dirs)}")

# 统计空目录和非空目录
empty_dirs = []
non_empty_dirs = []
file_types = Counter()
total_size = 0

for d in dirs:
    dir_path = os.path.join(bundles_path, d)
    files = os.listdir(dir_path)
    
    if not files:
        empty_dirs.append(d)
    else:
        non_empty_dirs.append(d)
        for f in files:
            file_path = os.path.join(dir_path, f)
            if os.path.isfile(file_path):
                size = os.path.getsize(file_path)
                total_size += size
                ext = os.path.splitext(f)[1]
                file_types[ext] += 1

print(f"Empty directories: {len(empty_dirs)}")
print(f"Non-empty directories: {len(non_empty_dirs)}")
print(f"Total size: {total_size / (1024**3):.2f} GB")
print(f"\nFile type distribution:")
for ext, count in file_types.most_common():
    ext_str = ext or '(no extension)'
    print(f"  {ext_str}: {count}")

# 显示几个有内容的目录示例
print(f"\nSample non-empty directories (first 5):")
for d in non_empty_dirs[:5]:
    dir_path = os.path.join(bundles_path, d)
    files = os.listdir(dir_path)
    print(f"\n{d}:")
    for f in files[:10]:
        file_path = os.path.join(dir_path, f)
        size = os.path.getsize(file_path)
        print(f"  {f} ({size / (1024**2):.2f} MB)")
    if len(files) > 10:
        print(f"  ... and {len(files) - 10} more files")

# 保存空目录列表到文件
with open(r'F:\Dev\RD-Agent-main\empty_bundle_dirs.txt', 'w') as f:
    for d in empty_dirs:
        f.write(d + '\n')
print(f"\nEmpty directories list saved to: F:\\Dev\\RD-Agent-main\\empty_bundle_dirs.txt")
