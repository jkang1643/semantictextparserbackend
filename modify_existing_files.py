#!/usr/bin/env python3
"""
Script to modify existing segmentation files to remove token limits.
This modifies the existing files in-place to remove token limit checks.
"""

import re
import os
from pathlib import Path


def remove_token_limits_from_file(file_path: str):
    """Remove token limit checks from a segmentation file."""
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return False
    
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Backup original
    backup_path = file_path + '.backup'
    with open(backup_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"💾 Created backup: {backup_path}")
    
    # Remove token limit checks
    modifications = [
        # Remove token limit checks in segmentation loops
        (
            r'# Check token limit first \(hard boundary\)\s*\n\s*if current_tokens \+ sentence_tokens > self\.max_tokens_per_chunk and current_scene:\s*\n\s*scenes\.append\(\' \'\.join\(current_scene\)\)\s*\n\s*current_scene = \[sentence\]\s*\n\s*current_tokens = sentence_tokens\s*\n\s*scene_context = \{[^}]+\}\s*\n\s*continue\s*\n',
            '# Token limit check removed - scenes can be any length\n'
        ),
        # Remove token limit checks in rule-based segmentation
        (
            r'# If adding this sentence would exceed the limit, start a new chunk\s*\n\s*if current_tokens \+ sentence_tokens > self\.max_tokens_per_chunk and current_chunk:\s*\n\s*chunks\.append\(\' \'\.join\(current_chunk\)\)\s*\n\s*current_chunk = \[sentence\]\s*\n\s*current_tokens = sentence_tokens\s*\n\s*else:\s*\n\s*current_chunk\.append\(sentence\)\s*\n\s*current_tokens \+= sentence_tokens',
            '# Token limit check removed - chunks can be any length\n            current_chunk.append(sentence)\n            current_tokens += sentence_tokens'
        ),
        # Update class docstrings
        (
            r'max_tokens_per_chunk: int = 512',
            'max_tokens_per_chunk: int = 10000  # High limit, not used for segmentation'
        ),
        # Update comments
        (
            r'Maximum tokens per chunk for LLM compatibility',
            'Maximum tokens per chunk (not used for segmentation)'
        )
    ]
    
    modified_content = content
    changes_made = 0
    
    for pattern, replacement in modifications:
        new_content = re.sub(pattern, replacement, modified_content, flags=re.MULTILINE | re.DOTALL)
        if new_content != modified_content:
            changes_made += 1
            modified_content = new_content
    
    # Write modified content
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(modified_content)
    
    if changes_made > 0:
        print(f"✅ Modified {file_path}: {changes_made} changes made")
        return True
    else:
        print(f"ℹ️  No changes needed in {file_path}")
        return False


def main():
    """Modify existing segmentation files to remove token limits."""
    
    files_to_modify = [
        'scene_segmenter.py',
        'intelligent_scene_segmenter.py', 
        'final_scene_segmenter.py',
        'scene_segmenter_v2.py',
        'text_segmenter.py',
        'text_segmenter_lite.py'
    ]
    
    print("🔧 Removing token limits from segmentation files...")
    print("=" * 50)
    
    modified_files = []
    for file_path in files_to_modify:
        if os.path.exists(file_path):
            if remove_token_limits_from_file(file_path):
                modified_files.append(file_path)
        else:
            print(f"⚠️  File not found: {file_path}")
    
    print("\n" + "=" * 50)
    print("📊 SUMMARY:")
    print(f"✅ Modified {len(modified_files)} files")
    
    if modified_files:
        print("\nModified files:")
        for file_path in modified_files:
            print(f"  - {file_path}")
    
    print("\n💡 Next steps:")
    print("1. Test the modified files with your text")
    print("2. If issues occur, restore from backup files (*.backup)")
    print("3. Consider using the new unlimited pipeline for better control")


if __name__ == "__main__":
    main()
