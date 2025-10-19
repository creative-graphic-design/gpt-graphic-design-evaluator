# GPT Graphic Design Evaluator

[![arXiv](https://img.shields.io/badge/arXiv-2410.08885-b31b1b.svg)](https://arxiv.org/abs/2410.08885)
[![arXiv](https://img.shields.io/badge/SIGGRAPH%20Asia-2024-black.svg)](https://dl.acm.org/doi/10.1145/3681758.3698010)

A [LangChain](https://www.langchain.com) implementation of an GPT-powered graphic design evaluation system based on the research paper [arXiv:2410.08885](https://arxiv.org/abs/2410.08885). This library uses large language models (LLMs) with vision capabilities to automatically assess graphic designs according to established design principles.

## Overview

The GPT Graphic Design Evaluator leverages the visual understanding capabilities of modern AI models to provide objective, structured feedback on graphic design quality. The system evaluates designs across three fundamental design principles:

- **Alignment**: Proper positioning and organization of design elements
- **Overlap**: Appropriate use of overlapping elements and visual layering
- **Whitespace**: Effective use of negative space and element spacing

## Installation

```bash
pip install git+https://github.com/creative-graphic-design/gpt-graphic-design-evaluator.git
pip install langchain-openai # or langchain-aws or some other LLM wrapper
```

## Quick Start

```python
from PIL import Image
from langchain_openai import ChatOpenAI  # or your preferred LLM
from gpt_graphic_design_evaluator import GPTGraphicDesignEvaluator

# Initialize the evaluator with your LLM
llm_model = ChatOpenAI(model="gpt-4o")  # Requires vision capabilities
evaluator = GPTGraphicDesignEvaluator(llm=llm_model)

# Load your rasterized design image
image = Image.open("your_design.png")

# Evaluate the design
result = evaluator(image, design_principle="alignment")  # or "overlap" or "whitespace"

print(f"Score: {result.score}/10")
print(f"Explanation: {result.explanation}")
```

## Advanced Usage

### Custom Evaluation Criteria

```python
# Use custom design principle prompt
custom_prompt = """\
Evaluate the color harmony and contrast in this design...
"""
result = evaluator.evaluate(
    image=image,
    design_principle_prompt=custom_prompt
)
```

### Custom System Prompt

```python
# Use custom system prompt template
custom_system_prompt = """\
You are a professional design critic. Evaluate this design with focus on {design_principle}.
Provide scores from 1-10 where 10 is exceptional.

Output format:
{{"score": X, "explanation": "Your detailed critique here"}}
"""

result = evaluator.evaluate(
    image=image,
    design_principle_prompt="focus on typography and readability",
    system_prompt_template=custom_system_prompt
)
```

## Acknowledgements

- CyberAgentAILab/Graphic-design-evaluation: This is the official repository for "Can GPTs Evaluate Graphic Design Based on Design Principles?". https://github.com/CyberAgentAILab/Graphic-design-evaluation 
