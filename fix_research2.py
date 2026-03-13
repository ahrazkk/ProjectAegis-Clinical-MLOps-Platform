import io

file_path = "src/pages/ResearchPage.jsx"
with io.open(file_path, "r", encoding="utf-8") as f:
    text = f.read()

# Make sure it isn't already there:
if "activeTab === 'evolution'" in text:
    print("Evolution block already exists! Searching and destroying before adding...")
    # Find start and end to remove
    start_str = "{/* START MODEL EVOLUTION TAB */}"
    end_str = "{/* END MODEL EVOLUTION TAB */}"
    s = text.find(start_str)
    e = text.find(end_str)
    if s != -1 and e != -1:
        text = text[:s] + text[e + len(end_str):]
    else:
        # just in case it was inserted without markers
        s_evo = text.find("{activeTab === 'evolution' && (")
        if s_evo != -1:
            print("Please fix manually, ambiguous start/end")

with io.open("inject3.py", "r", encoding="utf-8-sig") as f:
    inject3_text = f.read()

# We need everything from `new_insert = """` onward.
idx1 = inject3_text.find('new_insert = """') + len('new_insert = """')
idx2 = inject3_text.rfind('"""')
evolution_code = inject3_text[idx1:idx2]

idx_target = text.find("{activeTab === 'gnn' && (")
if idx_target != -1:
    print("Found insertion point.")
    text = text[:idx_target] + "{/* START MODEL EVOLUTION TAB */}\n            " + evolution_code + "\n            {/* END MODEL EVOLUTION TAB */}\n            " + text[idx_target:]
    with io.open(file_path, "w", encoding="utf-8") as f:
        f.write(text)
    print("Saved modified ResearchPage.jsx")
else:
    print("Could not find GNN tab insertion point")
