file_path = r"C:\Users\1kibr\Documents\WebDevelopment\DDI_PROJECTV2-FRONTEND\molecular-ai\src\pages\ResearchPage.jsx"
with open(file_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

# delete lines 329 to 463 (0-indexed: 328 to 463)
# the element at index 327 is END MODEL EVOLUTION TAB
del lines[328:463]

with open(file_path, "w", encoding="utf-8") as f:
    f.writelines(lines)
print("Deleted redundant lines.")
