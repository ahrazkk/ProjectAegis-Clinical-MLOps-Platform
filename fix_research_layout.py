import io

file_path = "src/pages/ResearchPage.jsx"

with io.open(file_path, "r", encoding="utf-8") as f:
    text = f.read()

# Fix the main wrapper
old_wrapper_cls = 'className="grid md:grid-cols-1 gap-12 relative before:absolute before:inset-0 before:ml-[23px] md:before:ml-1/2 before:-translate-x-px md:before:translate-x-0 before:w-0.5 before:bg-gradient-to-b before:from-purple-500/50 before:via-blue-500/50 before:to-cyan-400/50"'
new_wrapper_cls = 'className="grid md:grid-cols-1 gap-12 relative before:absolute before:inset-0 before:ml-[23px] before:w-0.5 before:bg-gradient-to-b before:from-purple-500/50 before:via-blue-500/50 before:to-cyan-400/50"'
text = text.replace(old_wrapper_cls, new_wrapper_cls)

# Fix Gen 1 card
old_card1_cls = 'className="relative flex items-center justify-between md:justify-normal md:odd:flex-row-reverse group border border-purple-500/20 bg-black/40 p-8 rounded-2xl hover:border-purple-500/50 transition-colors ml-12 md:ml-0 md:w-[calc(50%-3rem)] left-0"'
new_card1_cls = 'className="relative flex items-center group border border-purple-500/20 bg-black/40 p-8 rounded-2xl hover:border-purple-500/50 transition-colors ml-12 lg:ml-16 mr-0 w-full lg:w-[calc(100%-4rem)]"'
text = text.replace(old_card1_cls, new_card1_cls)

# Fix Gen 2 card
old_card2_cls = 'className="relative flex items-center justify-between md:justify-normal md:odd:flex-row-reverse group border border-blue-500/20 bg-black/40 p-8 rounded-2xl hover:border-blue-500/50 transition-colors ml-12 md:ml-0 md:w-[calc(50%-3rem)] md:left-1/2 md:translate-x-[3rem]"'
new_card2_cls = 'className="relative flex items-center group border border-blue-500/20 bg-black/40 p-8 rounded-2xl hover:border-blue-500/50 transition-colors ml-12 lg:ml-16 mr-0 w-full lg:w-[calc(100%-4rem)]"'
text = text.replace(old_card2_cls, new_card2_cls)

# Fix Gen 3 card
old_card3_cls = 'className="relative flex items-center justify-between md:justify-normal group border border-cyan-400/50 bg-cyan-950/10 p-8 md:p-12 rounded-2xl shadow-[0_0_40px_-10px_rgba(34,211,238,0.15)] ml-12 md:ml-0 md:w-[calc(50%-3rem)] left-0 border-l-4"'
new_card3_cls = 'className="relative flex items-center group border border-cyan-400/50 bg-cyan-950/10 p-8 md:p-12 rounded-2xl shadow-[0_0_40px_-10px_rgba(34,211,238,0.15)] ml-12 lg:ml-16 border-l-4 mr-0 w-full lg:w-[calc(100%-4rem)]"'
text = text.replace(old_card3_cls, new_card3_cls)

with io.open(file_path, "w", encoding="utf-8") as f:
    f.write(text)

print("Updated ResearchPage.jsx")
