import io

file_path = "src/pages/LandingPageV2.jsx"
with io.open(file_path, "r", encoding="utf-8") as f:
    text = f.read()

# Fix stats
text = text.replace("{ value: '92.7', label: 'AUC Score', suffix: '%' }", "{ value: '98.6', label: 'AUC Score', suffix: '%' }")
text = text.replace("{ value: '1.6K+', label: 'Interactions', suffix: '' }", "{ value: '53.4K+', label: 'Interactions', suffix: '' }")
text = text.replace("with 92.7% AUC accuracy", "with 98.67% AUC accuracy")

# Fix descriptions
text = text.replace("Future: Graph Neural Networks", "Production: GraphSAGE Neural Networks")
text = text.replace("Roadmap includes GNN architecture for molecular graph processing and enhanced accuracy", "Macroscopic GraphSAGE architecture processing 53k+ clinical pathways with 98.67% precision")

text = text.replace("92.7% AUC</span> on the DDI Corpus 2013 benchmark", "98.67% AUC</span> on the TWOSIDES global interactome benchmark")

with io.open(file_path, "w", encoding="utf-8") as f:
    f.write(text)

file_path_old = "src/pages/LandingPage.jsx"
with io.open(file_path_old, "r", encoding="utf-8") as f:
    text2 = f.read()

text2 = text2.replace("Future: Graph Neural Networks", "Production: GraphSAGE Neural Networks")
text2 = text2.replace("Roadmap includes GNN architecture for molecular graph processing and enhanced accuracy", "Macroscopic GraphSAGE architecture processing 53k+ clinical pathways with 98.67% precision")

with io.open(file_path_old, "w", encoding="utf-8") as f:
    f.write(text2)

print("Updated Landing Pages")