from long_term_memory import memory_manager


texts = [
    "I like going outside",
    "I hate my classmates",
    "I'm out of coffee",
    "It's my birthday !",
    "I'm allergic to coffee",
    "I'd like to go to the mountains",
    "I'm fucking sick of this life"
]
for text in texts: memory_manager.add_new_memory(text)