import os
from collections import Counter

bundles_path = r'F:\Dev\RD-Agent-main\RDagentDB\production_bundles'

# 分析文件命名模式
file_patterns = Counter()
file_sizes = []

for d in os.listdir(bundles_path):
    dir_path = os.path.join(bundles_path, d)
    if not os.path.isdir(dir_path):
        continue
    
    for f in os.listdir(dir_path):
        file_path = os.path.join(dir_path, f)
        if os.path.isfile(file_path):
            size = os.path.getsize(file_path)
            file_sizes.append(size)
            
            # 提取文件类型（去掉前缀hash）
            parts = f.split('_', 1)
            if len(parts) > 1:
                file_type = parts[1]
                file_patterns[file_type] += 1

print("File type patterns (after hash prefix):")
for pattern, count in file_patterns.most_common(20):
    print(f"  {pattern}: {count}")

# 统计文件大小分布
file_sizes.sort()
total_size = sum(file_sizes)
print(f"\nTotal files: {len(file_sizes)}")
print(f"Total size: {total_size / (1024**3):.2f} GB")
print(f"Average file size: {total_size / len(file_sizes) / (1024**2):.2f} MB")
print(f"Median file size: {file_sizes[len(file_sizes)//2] / (1024**2):.2f} MB")

# 按大小分类
small_files = [s for s in file_sizes if s < 1024*1024]  # < 1MB
medium_files = [s for s in file_sizes if 1024*1024 <= s < 10*1024*1024]  # 1-10MB
large_files = [s for s in file_sizes if 10*1024*1024 <= s < 100*1024*1024]  # 10-100MB
huge_files = [s for s in file_sizes if s >= 100*1024*1024]  # >= 100MB

print(f"\nSize distribution:")
print(f"  Small (<1MB): {len(small_files)} files, {sum(small_files)/(1024**3):.2f} GB")
print(f"  Medium (1-10MB): {len(medium_files)} files, {sum(medium_files)/(1024**3):.2f} GB")
print(f"  Large (10-100MB): {len(large_files)} files, {sum(large_files)/(1024**3):.2f} GB")
print(f"  Huge (>=100MB): {len(huge_files)} files, {sum(huge_files)/(1024**3):.2f} GB")

# 分析备份文件
backup_files = []
for d in os.listdir(bundles_path):
    dir_path = os.path.join(bundles_path, d)
    if not os.path.isdir(dir_path):
        continue
    
    for f in os.listdir(dir_path):
        if 'backup' in f.lower() or '.bak' in f.lower():
            file_path = os.path.join(dir_path, f)
            if os.path.isfile(file_path):
                backup_files.append((d, f, os.path.getsize(file_path)))

print(f"\nBackup files found: {len(backup_files)}")
backup_size = sum(size for _, _, size in backup_files)
print(f"Backup files total size: {backup_size / (1024**2):.2f} MB")

if backup_files:
    print("\nSample backup files:")
    for d, f, size in backup_files[:10]:
        print(f"  {d}/{f} ({size/1024:.2f} KB)")
