from typing import *
import argparse
import os
from shutil import copyfile

parser = argparse.ArgumentParser()
parser.add_argument('--input-dir', type=str)
parser.add_argument('--dev', type=str)
parser.add_argument('--test', type=str)
parser.add_argument('--output-dir', type=str)
args = parser.parse_args()

all_files: List[str] = [fn for fn in os.listdir(args.input_dir) if '.concrete' in fn]

test_files: List[str] = []
with open(args.test) as f:
    for line in f:
        test_files.append(f'{line.strip()}.concrete')

dev_files: List[str] = []
with open(args.dev) as f:
    for line in f:
        dev_files.append(f'{line.strip()}.concrete')

train_files: List[str] = list(set(all_files) - set(test_files) - set(dev_files))

assert len(set(train_files) & set(dev_files)) == 0
assert len(set(dev_files) & set(test_files)) == 0
assert len(set(train_files) & set(test_files)) == 0

train_dir: str = os.path.join(args.output_dir, 'train')
dev_dir: str = os.path.join(args.output_dir, 'dev')
test_dir: str = os.path.join(args.output_dir, 'test')

os.makedirs(train_dir, exist_ok=True)
os.makedirs(dev_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# old dataset
test_files: List[str] = list(set(test_files) & set(all_files))
for fn in train_files:
    copyfile(os.path.join(args.input_dir, fn), os.path.join(train_dir, fn))
for fn in dev_files:
    copyfile(os.path.join(args.input_dir, fn), os.path.join(dev_dir, fn))
for fn in test_files:
    copyfile(os.path.join(args.input_dir, fn), os.path.join(test_dir, fn))
