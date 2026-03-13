# -*- coding: utf-8 -*-
import re

file_path = r"C:\Users\1kibr\Documents\WebDevelopment\DDI_PROJECTV2-FRONTEND\molecular-ai\src\pages\ResearchPage.jsx"
with open(file_path, "r", encoding="utf-8") as f:
    code = f.read()

# We need to find wherever END MODEL EVOLUTION TAB is, and then find GNN RESEARCH TAB, and delete everything between them.
# The user's original commented block for GNN RESEARCH TAB has unicode characters.
# We'll use regex to match the comment containing GNN RESEARCH TAB

pattern = re.compile(r'\{\/\*\s*END MODEL EVOLUTION TAB\s*\*\/\}.*?(?=\{\/\*.*?GNN RESEARCH TAB.*?\*\/\})', re.DOTALL)

res = pattern.search(code)
if res:
    print("Found the duplicate block, removing it.")
    new_code = pattern.sub("{/* END MODEL EVOLUTION TAB */}\n\n          ", code)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(new_code)
else:
    print("Duplicate block not found.")

