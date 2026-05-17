from truthfulqa_bench.dataset import TruthfulQARow
from truthfulqa_bench.prompting import build_binary_prompt, build_two_order_prompts, parse_choice


ROW = TruthfulQARow(
    row_id=7,
    category="Fiction",
    question="Which answer is true?",
    best_answer="The truthful answer.",
    best_incorrect_answer="The false answer.",
)


def test_build_binary_prompt_correct_first():
    prompt = build_binary_prompt(ROW, "correct_first")

    assert prompt.correct_choice == "A"
    assert "A. The truthful answer." in prompt.prompt
    assert "B. The false answer." in prompt.prompt


def test_build_binary_prompt_incorrect_first():
    prompt = build_binary_prompt(ROW, "incorrect_first")

    assert prompt.correct_choice == "B"
    assert "A. The false answer." in prompt.prompt
    assert "B. The truthful answer." in prompt.prompt


def test_build_two_order_prompts_reverses_answers():
    first, second = build_two_order_prompts(ROW)

    assert first.order == "correct_first"
    assert second.order == "incorrect_first"
    assert first.correct_choice == "A"
    assert second.correct_choice == "B"


def test_parse_choice_is_strict():
    assert parse_choice(" A ") == "A"
    assert parse_choice("b") == "B"
    assert parse_choice("Answer: A") is None
    assert parse_choice("AB") is None
