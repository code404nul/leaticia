from long_term_memory import final_score

query = "I'm going to buy some coffee"

scores, outputs = final_score.similarity_total("I'm going to buy some coffee")
final_score.show_resut_similarity(outputs, scores, query)