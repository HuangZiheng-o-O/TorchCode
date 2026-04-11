import os

def main():
    dir_name = 'ML面试问题'
    readme_name = 'README_my_面试.md'
    
    # 物理文件映射 (原文件名 -> 目标文件名部分)
    # 格式: (类别, 章节序号, TAG, [文件名列表])
    mapping = [
        ("01", "BASIC", [
            "ChatGPT-Adam vs SGD.md",
            "ChatGPT-Dropout Weight Initialization Gradient Accumulation.md",
            "ChatGPT-KL 散度解释.md",
            "ChatGPT-交叉熵与KL散度.md",
            "ChatGPT-偏差方差权衡解析.md",
            "ChatGPT-梯度问题与梯度裁剪.md",
            "ChatGPT-激活函数对比.md",
            "ChatGPT-过拟合及其缓解方法.md"
        ]),
        ("02", "TRANSFORMER", [
            "ChatGPT-Causal vs Cross Attention.md",
            "ChatGPT-FlashAttention 瓶颈解决.md",
            "ChatGPT-FlashAttention深度讲解.md",
            "ChatGPT-KV Cache 解析.md",
            "ChatGPT-Norm 选择与 LLM.md",
            "ChatGPT-RoPE 位置编码解析.md",
            "ChatGPT-Self-attention 公式与复杂度.md",
            "ChatGPT-Transformer 与 RNN 比较.md",
            "ChatGPT-多头注意力原理解释.md"
        ]),
        ("03", "SFT", [
            "ChatGPT-LoRA QLoRA 全量微调选择.md",
            "ChatGPT-RAG vs Fine-Tuning.md"
        ]),
        ("04", "ALIGN", [
            "ChatGPT-Branch · DPO PPO GRPO Loss.md",
            "ChatGPT-DPO PPO GRPO Loss.md"
        ]),
        ("05", "INFER", [
            "ChatGPT-LLM系统评估方法.md",
            "ChatGPT-ML AI LLM面试题.md",
            "ChatGPT-生成策略比较.md",
            "ChatGPT-量化影响分析.md"
        ])
    ]

    readme_content = "# 🧠 ML/LLM 面试题库 (Obsidian 版)\n\n"
    readme_content += "> 按面试逻辑分类，点击 [[双链]] 即可直接跳转。\n\n"

    for chap_id, tag, files in mapping:
        readme_content += f"### {chap_id} {tag}\n\n"
        readme_content += "| # | 面试题目 | 知识点 | 难度 |\n"
        readme_content += "| :---: | :--- | :--- | :---: |\n"
        
        for idx, old_name in enumerate(files, 1):
            # 提取原中文名 (去掉 ChatGPT- 和 .md)
            clean_name = old_name.replace("ChatGPT-", "").replace(".md", "")
            
            # 构建新文件名: 章节序号-相对序号-TAG-名称.md
            new_name_base = f"{chap_id}-{idx:02d}-{tag}-{clean_name}"
            new_filename = f"{new_name_base}.md"
            
            # 物理重命名
            old_path = os.path.join(dir_name, old_name)
            new_path = os.path.join(dir_name, new_filename)
            
            if os.path.exists(old_path):
                os.rename(old_path, new_path)
                print(f"Renamed: {old_name} -> {new_filename}")
            else:
                print(f"Skipped (not found): {old_name}")
            
            # 写入 README (Obsidian 格式: [[文件名]])
            readme_content += f"| {idx:02d} | [[{new_name_base}]] | {tag} | ⭐ |\n"
        
        readme_content += "\n"

    with open(readme_name, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    print(f"\nSuccessfully created {readme_name}")

if __name__ == "__main__":
    main()
