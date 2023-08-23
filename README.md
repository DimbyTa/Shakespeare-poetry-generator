# Shakespeare-poetry-generator

A GRU character-based language model, trained on Shakespeare's poems from https://allpoetry.com to generate poems, implemented using Pytorch. 
This work is inspired by the seventh chapter of the book Natural Language Processing with Pytorch by Delip Rao and Brian McMahan.

# Files organization
* shakespeare-scraper.ipynb: the Spider scraper used to scrape the poems.
* cleaning-poems.ipynb: basic cleaning of extra space, tabulation, newline characters, dash and double dash.
* shakespeare-generator.ipynb: the model implemented in Pytorch.
Those notebooks were executed on Kaggle.

# Challenges
* Poems are long sequences of characters, there is a risk of running into vanishing gradient, rendering the training inefficient and the generated poems unreadable.
* Adding seed word to the generation process: differently from the unconditioned model presented in the book, we add a seed word to condition the output in some way. But the addition of a seed word complicated the handling of the shape of the hidden state expected by the GRU unit, it expected a hidden state of shape (number_directions * number_layers, number_poems, hidden_state_size), and the seed word tensor was of shape: (number_directions * number_layers, number_poems, length_seed_word).

# Proposed solutions
* The sequences were subdivided into overlapping subsequences of length subsequence_length. The amout of overlap is controled using the stride parameter.
* In order to match the shape of the hidden state, the seed word was padded using padding character, i.e: until length_seed_world matches hidden_state_size, we fill the seed word with padding character.

# Next steps
* Evaluate the goodness of the outputs. Create a metric based on correct words, grammatical correctness first, and extend to the correctness of the subject.
* Fine-tuning: fine tune the hyperparameters to maximize the score previously discussed.
* Conditioning the outputs: scrape more poems, and group by author and/or subject. Condition the outputs based on author and/or subject.
