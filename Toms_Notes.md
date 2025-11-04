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

---

## Mid-Generation Perturbation via Conversation History (Technical Solution)

**Key Insight**: Closed-source LLM APIs (OpenAI, Anthropic, Google, etc.) allow you to provide conversation history including "assistant" role messages. The model treats these assistant messages as if it had generated them itself.

**How It Works**:
```
messages = [
    {"role": "user", "content": "Explain quantum computing"},
    {"role": "assistant", "content": "Quantum computing uses..."},  # We inject this!
]
# Model continues from this point as if it wrote that text
```

**Experimental Design**:
1. Generate full response to complex question (control condition)
2. Keep first half of model output, split at sentence boundary
3. Modify one sentence in the first half using sentence-BERT paraphrasing
4. Inject modified first half as "assistant" message in conversation history
5. Request continuation from that point
6. Compare original second half vs. new continuation (measure divergence)

**Implementation Requirements**:
- Add `generate_with_context(messages)` method to model wrappers
- Messages array contains full conversation history with roles
- Works identically across OpenAI, Anthropic, Google APIs

**Why This Matters**: 
This enables testing whether semantic perturbations in the model's OWN OUTPUT (not just input prompts) cause divergence in subsequent generation. Tests stability of the generation process itself as an iterative system.

**Experimental Variations**:
- Modify different positions (early/middle/late in first half)
- Vary perturbation magnitude (small paraphrase vs. semantic shift)
- Test with factual vs. creative vs. reasoning tasks
- Compare models: Do some recover better from perturbed context?
