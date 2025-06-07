from long_term_memory.FAISS import SemanticSearchEngine

engine = SemanticSearchEngine()

query = "i love to go hiking"
results = engine.search(query, seuil=0.45)  # baisse un peu le seuil pour être sûr
engine.pretty_print_results(query, results)
