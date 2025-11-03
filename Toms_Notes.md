Have a serious look at @README.md as a chief science officer.  Before proceeding with this project I'd like to think about the overall goals and revise next steps.  I want to measure the semantic divergence rate of open and closed models under different scenarios.  LLM outputs are a sequential process and therefore could potentially accumulate error and diverge. 

We could have multiple sentence prompts and only change one setence using our sentence-BERT method applied to that sentence.  However, sentence-BERT woudn't necessarily have the conext of the rest of the prompt, so we could have a strong AI judge if the two paragraphs are different and if the altered sentence changes the meaning (as a check).  But then we need to check the semantics of the output too.  

Or we could test it on a verifiable math or coding problem.

--> How can we simulate a chat dialog and change one sentence in the middle?  
- Find a sentence that could have only one meaning and change it with setence Bert.

Can we test the stability versus the size of the input prompt?

Can we test vs temperature?

with and without reasoning or thinking.

Ask a problem which takes a lot of thinking, but only verify the semantics of the final one sentence answer.  Like jeopardy almost.  The semantics of a single sentence can be readily measured and we want to know if thinking improves stability.  

Or do a task which involves more generative exploration.  Would a math AI with neutral stability do better than a highly convergent one?  And how does temperature play in?

