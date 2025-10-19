import base64
import io
from dataclasses import dataclass
from typing import Dict, Final, Literal, Optional, cast

from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from langchain_core.language_models import BaseChatModel
from PIL import Image
from pydantic import BaseModel, Field

DesignPrinciple = Literal["alignment", "overlap", "whitespace"]

SYSTEM_PROMPT: Final[str] = """\
You are an autonomous AI Assistant who aids designers by providing insightful, objective, and constructive critiques of graphic design projects. Your goals are: "Deliver comprehensive and unbiased evaluations of graphic designs based on the following design principles."

Grade seriously. The range of scores is from 1 to 10. A flawless design can earn 10 points, a mediocre design can only earn 7 points, a design with obvious shortcomings can only earn 4 points, and a very poor design can only earn 1-2 points.

{design_principle}

If the output is too long, it will be truncated. Only respond in JSON format, no other information. Example of output for a better graphic design:

{{
    "score": 6, 
    "explanation": "Please concisely explain the reason of the score."
}}"""


ALIGNMENT_DESIGN_PRINCIPLE: Final[str] = """\
Correct alignment is an important aspect of design that has been modeled in other layout applications. Text and graphic elements are aligned on the page to indicate organizational structure and aesthetics.

Please evaluate the alignment of the input graphic design considering the following points.

1. Alignment along with the horizontal and vertical direction is considered.
2. The elements that align at a glance but slight misalignment are penalized because it is visually displeasing.
3. Larger alignment groups (i.e., aligned elements that are distant from each other) are preferred as they produce simpler designs with more unity between elements."""

OVERLAP_DESIGN_PRINCIPLE: Final[str] = """\
Overlapping elements are common in many designs and absent from others.
Less or proper overlapping might be considered aesthetically pleasing, but others are not.

Please consider the following points to evaluate the overlap.

1. The three types of overlap, the overlap of elements on text, the overlap of text on graphics, and the overlap of graphics on other graphics, are considered.
2. Hard-to-read text because of insufficient color contrast between a text and the background color is penalized.
3. The graphic design that includes elements extending past the boundaries is also penalized."""

WHITE_SPACE_DESIGN_PRINCIPLE: Final[str] = """\
White space in graphic designs is fundamental for readability and aesthetics. Element distance is also closely related to the principle of proximity, as elements placed near each other may appear to be related. White space also influences the overall design style; many modern designs use significant white space. White space 'trapped' between elements can also be distracting. 

Evaluate the white space considering the following points.

1.A large ratio of white space that is not covered by design elements (e.g., graphics and tests) is preferred.
2. However, the graphic design with a too large region of empty white space on the image is undesirable.
3. The greater the distance between each element is preferred.
4. Uniformed vertical spacing of each text element is preferred.
5. Wider border margins for each element are preferred."""


DESIGN_PRINCIPLES: Dict[DesignPrinciple, str] = {
    "alignment": ALIGNMENT_DESIGN_PRINCIPLE,
    "overlap": OVERLAP_DESIGN_PRINCIPLE,
    "whitespace": WHITE_SPACE_DESIGN_PRINCIPLE,
}

USER_PROMPT: Final[str] = """\
Please score the following images."""


class EvaluationResult(BaseModel):
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
class GPTGraphicDesignEvaluator(object):
    llm: BaseChatModel

    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert a PIL Image to a base64-encoded PNG string.

        Args:
            image (Image.Image): The input PIL Image.

        Returns:
            str: The base64-encoded PNG string.
        """
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_str

    def evaluate(
        self,
        image: Image.Image,
        design_principle_prompt: str,
        system_prompt_template: Optional[str] = None,
    ) -> EvaluationResult:
        """Evaluate the graphic design image based on the given design principle.

        Args:
            image (Image.Image): The input graphic design image as a PIL Image.
            design_principle_prompt (str): The design principle prompt to guide the evaluation.
            system_prompt_template (Optional[str], optional): The system prompt template.
                If None, the default SYSTEM_PROMPT is used. Defaults to None.

        Returns:
            EvaluationResult: The evaluation result containing the score and explanation.
        """
        system_prompt_template = system_prompt_template or SYSTEM_PROMPT

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

        chain = prompt | self.llm.with_structured_output(EvaluationResult)

        input_data = {
            "design_principle": design_principle_prompt,
            "base64_image": self._image_to_base64(image),
        }
        return cast(EvaluationResult, chain.invoke(input_data))

    def __call__(
        self, image: Image.Image, design_principle: DesignPrinciple
    ) -> EvaluationResult:
        """Evaluate the graphic design image based on the specified design principle.

        Args:
            image (Image.Image): The input graphic design image as a PIL Image.
            design_principle (DesignPrinciple): The design principle to evaluate.

        Returns:
            EvaluationResult: The evaluation result containing the score and explanation.
        """
        return self.evaluate(
            image=image,
            system_prompt_template=SYSTEM_PROMPT,
            design_principle_prompt=DESIGN_PRINCIPLES[design_principle],
        )
