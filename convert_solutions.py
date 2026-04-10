import os
import json
import glob

def convert_ipynb_to_md(ipynb_path, md_path):
    with open(ipynb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    md_content = []
    for cell in nb.get('cells', []):
        cell_type = cell.get('cell_type')
        source = cell.get('source', [])
        if isinstance(source, list):
            source = "".join(source)
        
        if cell_type == 'markdown':
            md_content.append(source)
        elif cell_type == 'code':
            md_content.append(f"```python\n{source}\n```")
    
    # 写入时默认保留了 cell 之间的空行，如需彻底去掉空行，可在此处处理
    final_text = "\n\n".join(md_content)
    
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(final_text)

def main():
    solutions_dir = 'solutions'
    output_dir = 'solutions_markdown'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    ipynb_files = glob.glob(os.path.join(solutions_dir, '*.ipynb'))
    for ipynb_file in ipynb_files:
        base_name = os.path.basename(ipynb_file)
        md_name = base_name.replace('.ipynb', '.md')
        md_path = os.path.join(output_dir, md_name)
        print(f"Converting {ipynb_file} to {md_path}")
        convert_ipynb_to_md(ipynb_file, md_path)

if __name__ == "__main__":
    main()
