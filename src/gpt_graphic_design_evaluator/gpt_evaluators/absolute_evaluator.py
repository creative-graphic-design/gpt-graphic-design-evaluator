from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, List, Optional, cast

from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from langchain_core.language_models import BaseChatModel
from PIL import Image
from pydantic import BaseModel, Field

from ..utils import image_to_base64
from .prompts import DESIGN_PRINCIPLES

if TYPE_CHECKING:
    from .prompts import DesignPrinciple


DEFAULT_SYSTEM_PROMPT: Final[str] = """\
You are an autonomous AI Assistant who aids designers by providing insightful, objective, and constructive critiques of graphic design projects. Your goals are: "Deliver comprehensive and unbiased evaluations of graphic designs based on the following design principles."

Grade seriously. The range of scores is from 1 to 10. A flawless design can earn 10 points, a mediocre design can only earn 7 points, a design with obvious shortcomings can only earn 4 points, and a very poor design can only earn 1-2 points.

{design_principle}

If the output is too long, it will be truncated. Only respond in JSON format, no other information. Example of output for a better graphic design:

{{
    "score": 6, 
    "explanation": "Please concisely explain the reason of the score."
}}"""


USER_PROMPT: Final[str] = """\
Please score the following images."""


class AbsoluteEvaluationResult(BaseModel):
    """Evaluation result model for graphic design evaluation."""

    score: int = Field(
        ge=1,
        le=10,
        description="Score from 1 to 10.",
    )
    explanation: str = Field(
        description="Concise explanation of the reason of the score."
    )


@dataclass
class GPTGraphicDesignAbsoluteEvaluator(object):
    llm: BaseChatModel

    def evaluate(
        self,
        image: Image.Image,
        design_principle_prompt: str,
        system_prompt_template: Optional[str] = None,
        num_return: int = 1,
    ) -> List[AbsoluteEvaluationResult]:
        """Evaluate the graphic design image based on the given design principle.

        Args:
            image (Image.Image): The input graphic design image as a PIL Image.
            design_principle_prompt (str): The design principle prompt to guide the evaluation.
            system_prompt_template (Optional[str], optional): The system prompt template.
                If None, the default SYSTEM_PROMPT is used. Defaults to None.
            num_return (int, optional): The number of evaluation results to return. Defaults to 1.

        Returns:
            EvaluationResult: The evaluation result containing the score and explanation.
        """
        system_prompt_template = system_prompt_template or DEFAULT_SYSTEM_PROMPT

        system_prompt = SystemMessagePromptTemplate.from_template(
            template=system_prompt_template,
        )
        user_prompt = (
            "user",
            [
                {"type": "text", "text": USER_PROMPT},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,{base64_image}"},
                },
            ],
        )
        prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt])

        chain = prompt | self.llm.with_structured_output(AbsoluteEvaluationResult)

        input_data = {
            "design_principle": design_principle_prompt,
            "base64_image": image_to_base64(image),
        }
        inputs = [input_data] * num_return

        return cast(List[AbsoluteEvaluationResult], chain.batch(inputs))

    def __call__(
        self, image: Image.Image, design_principle: DesignPrinciple, num_return: int = 1
    ) -> List[AbsoluteEvaluationResult]:
        """Evaluate the graphic design image based on the specified design principle.

        Args:
            image (Image.Image): The input graphic design image as a PIL Image.
            design_principle (DesignPrinciple): The design principle to evaluate.
            num_return (int, optional): The number of evaluation results to return. Defaults to 1.

        Returns:
            EvaluationResult: The evaluation result containing the score and explanation.
        """
        return self.evaluate(
            image=image,
            system_prompt_template=DEFAULT_SYSTEM_PROMPT,
            design_principle_prompt=DESIGN_PRINCIPLES[design_principle],
            num_return=num_return,
        )
