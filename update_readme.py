import re

def update_links(content):
    # Regex to find the pattern in the table: 
    # <a href="https://github.com/duoan/TorchCode/blob/master/templates/(\d+)_([\w-]+)\.ipynb" target="_blank">
    # And replace it with:
    # <a href="./solutions_markdown/\1_\2_solution.md" target="_blank">
    
    pattern = r'href="https://github\.com/duoan/TorchCode/blob/master/templates/(\d+)_([\w-]+)\.ipynb"'
    replacement = r'href="./solutions_markdown/\1_\2_solution.md"'
    
    updated_content = re.sub(pattern, replacement, content)
    return updated_content

def main():
    with open('README.md', 'r', encoding='utf-8') as f:
        content = f.read()
    
    updated_content = update_links(content)
    
    with open('README_my.md', 'w', encoding='utf-8') as f:
        f.write(updated_content)
    
    print("Successfully created README_my.md with updated links.")

if __name__ == "__main__":
    main()
