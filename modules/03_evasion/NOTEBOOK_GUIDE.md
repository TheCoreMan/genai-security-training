# Theory Notebooks Guide

## What to Expect

The theory notebooks (`adversarial_nlp.ipynb` and `certified_defenses.ipynb`) contain three types of content:

### 1. ✅ Fully Runnable Cells

**Setup & Configuration**:
- Cell 1: Device detection (CUDA/MPS/CPU)
- Cell 2: Package installation (auto-installs missing packages)

**Conceptual Examples**:
- Discrete vs continuous input space
- Semantic constraints demonstration
- Empirical vs certified defense comparison

**Working Examples** (NEW!):
- Homoglyph attack on real sentiment model
- Synonym substitution attack
- Simple randomized smoothing demonstration

These cells will run successfully and produce output!

### 2. 📚 Reference Implementations

**Class Definitions & Functions**:
- `class RandomizedSmoothing` - Full implementation
- `class IntervalBoundPropagation` - IBP implementation
- Various attack functions (HotFlip, TextFooler, etc.)

These cells define classes and functions but don't execute them. They're meant to:
- Show you how algorithms work
- Provide code you can copy/adapt
- Serve as reference implementations

**To use them**: Copy the code and integrate it into your own projects or the labs.

### 3. 📖 Optional Framework Examples (Markdown)

Some cells show how to use optional frameworks like:
- TextAttack
- OpenAttack
- Specialized libraries

These are shown as markdown code blocks with installation instructions. They're optional and require additional packages.

## How to Use These Notebooks

### Recommended Approach

1. **Run the setup cells** (1-2) to configure your environment
2. **Run the conceptual examples** to understand the ideas
3. **Run the working examples** to see attacks/defenses in action
4. **Study the reference implementations** to understand algorithms
5. **Copy code to labs** when you need specific implementations

### What NOT to Do

❌ Don't try to run every cell from top to bottom
- Some cells are reference implementations only
- Some require data/models not loaded in the notebook

✅ Instead: Run the working examples, study the implementations, use them in labs

## Cell-by-Cell Guide

### adversarial_nlp.ipynb

| Cells | Type | Can Run? | Purpose |
|-------|------|----------|---------|
| 1-2 | Setup | ✅ Yes | Device detection, packages |
| 3-11 | Conceptual | ✅ Yes | Understand NLP attack challenges |
| 12-15 | Working Examples | ✅ Yes | See real attacks in action |
| 16-60 | Reference | 📚 Study | Algorithm implementations |
| 61+ | Optional | 📖 Read | Framework examples |

### certified_defenses.ipynb

| Cells | Type | Can Run? | Purpose |
|-------|------|----------|---------|
| 1-2 | Setup | ✅ Yes | Device detection, packages |
| 3-10 | Conceptual | ✅ Yes | Understand certification |
| 11-13 | Working Example | ✅ Yes | See smoothing in action |
| 14-56 | Reference | 📚 Study | Defense implementations |

## Getting the Most Value

### For Learning
1. Read the explanations
2. Run the working examples
3. Study the reference implementations
4. Understand the algorithms

### For Projects
1. Copy reference implementations
2. Adapt to your use case
3. Test in the labs
4. Build your own attacks/defenses

### For Experimentation
1. Modify working examples
2. Try different parameters
3. Test on different models
4. Compare results

## Common Questions

**Q: Why don't all cells run?**
A: Some cells are reference implementations meant to be copied/adapted, not executed directly.

**Q: How do I use the reference implementations?**
A: Copy the code into your own scripts or the lab notebooks where you have models/data loaded.

**Q: What if I want to run everything?**
A: Check out the labs! They have complete end-to-end implementations with data and models.

**Q: Are the working examples enough?**
A: They demonstrate key concepts. For comprehensive practice, complete the labs.

## Next Steps

After working through these notebooks:
1. Complete the hands-on labs
2. Check your work against ANSWERS.ipynb
3. Experiment with your own attacks/defenses
4. Apply techniques to real-world problems
