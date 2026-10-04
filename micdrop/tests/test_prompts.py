"""The system prompts are the product definition; test them as such.

Each requirement from the spec has a matching assertion, so a future edit that
quietly drops "do not add information" from the REFINE prompt, or lets GENERATE
start apologising about the transformation, fails here rather than in the user's
editor.
"""

import unittest

from ..app.errors import UnknownModeError
from ..app.modes import MODE_CONTRACTS, Mode
from ..refinement.prompts import (
    GENERATE_SYSTEM_PROMPT,
    REFINE_SYSTEM_PROMPT,
    SYSTEM_PROMPTS,
    system_prompt_for,
)


class TestOnePromptPerMode(unittest.TestCase):
    def test_exactly_one_prompt_per_mode(self):
        self.assertEqual(set(SYSTEM_PROMPTS), set(Mode))

    def test_prompts_are_non_empty_and_distinct(self):
        for prompt in (REFINE_SYSTEM_PROMPT, GENERATE_SYSTEM_PROMPT):
            self.assertTrue(prompt.strip())
        self.assertNotEqual(REFINE_SYSTEM_PROMPT, GENERATE_SYSTEM_PROMPT)

    def test_lookup_returns_the_mode_own_prompt(self):
        self.assertEqual(system_prompt_for(Mode.REFINE), REFINE_SYSTEM_PROMPT)
        self.assertEqual(system_prompt_for(Mode.GENERATE), GENERATE_SYSTEM_PROMPT)

    def test_lookup_raises_for_anything_else(self):
        for bad in ("refine", None, 3, "auto"):
            with self.assertRaises(UnknownModeError):
                system_prompt_for(bad)

    def test_no_shared_fallback_exists(self):
        # A default entry would let a new mode silently inherit one prompt.
        self.assertNotIn(None, SYSTEM_PROMPTS)

    def test_each_prompt_names_its_own_mode(self):
        self.assertIn("REFINE", REFINE_SYSTEM_PROMPT)
        self.assertIn("GENERATE", GENERATE_SYSTEM_PROMPT)
        self.assertNotIn("GENERATE", REFINE_SYSTEM_PROMPT)
        self.assertNotIn("REFINE", GENERATE_SYSTEM_PROMPT)


class TestRefinePromptCoversItsRequirements(unittest.TestCase):
    def test_preserve_meaning(self):
        self.assertIn("Preserve meaning", REFINE_SYSTEM_PROMPT)

    def test_remove_filler(self):
        self.assertIn("filler words", REFINE_SYSTEM_PROMPT)

    def test_remove_repetitions(self):
        self.assertIn("repetitions", REFINE_SYSTEM_PROMPT)

    def test_correct_transcription_mistakes(self):
        self.assertIn("speech-recognition mistakes", REFINE_SYSTEM_PROMPT)

    def test_fix_grammar_and_punctuation(self):
        self.assertIn("grammar and punctuation", REFINE_SYSTEM_PROMPT)

    def test_preserve_technical_terminology(self):
        self.assertIn("technical terminology", REFINE_SYSTEM_PROMPT)

    def test_do_not_answer_the_request(self):
        self.assertIn("Do not answer", REFINE_SYSTEM_PROMPT)
        self.assertIn("not an email", REFINE_SYSTEM_PROMPT)

    def test_do_not_add_information(self):
        self.assertIn("Do not add information", REFINE_SYSTEM_PROMPT)

    def test_do_not_infer_missing_requirements(self):
        self.assertIn("Do not infer requirements", REFINE_SYSTEM_PROMPT)

    def test_do_not_expand(self):
        self.assertIn("Do not expand", REFINE_SYSTEM_PROMPT)
        self.assertIn("A short request stays short", REFINE_SYSTEM_PROMPT)

    def test_output_only_the_refined_text(self):
        self.assertIn("must be the refined text and nothing else", REFINE_SYSTEM_PROMPT)

    def test_no_reasoning_or_think_tags(self):
        self.assertIn("Do not output reasoning", REFINE_SYSTEM_PROMPT)
        self.assertIn("<think>", REFINE_SYSTEM_PROMPT)

    def test_stays_in_the_speakers_voice(self):
        self.assertIn("speaker's voice", REFINE_SYSTEM_PROMPT)


class TestGeneratePromptCoversItsRequirements(unittest.TestCase):
    def test_follow_the_instruction(self):
        self.assertIn("Follow the user's instruction", GENERATE_SYSTEM_PROMPT)

    def test_produce_the_artifact_directly(self):
        self.assertIn("Produce the artifact directly", GENERATE_SYSTEM_PROMPT)

    def test_does_not_discuss_the_transformation(self):
        self.assertIn("Do not discuss the transformation", GENERATE_SYSTEM_PROMPT)

    def test_does_not_expose_reasoning(self):
        self.assertIn("Do not expose reasoning", GENERATE_SYSTEM_PROMPT)
        self.assertIn("<think>", GENERATE_SYSTEM_PROMPT)

    def test_does_not_add_unrelated_content(self):
        self.assertIn("Do not add unrelated content", GENERATE_SYSTEM_PROMPT)

    def test_preserves_explicit_constraints(self):
        self.assertIn("Preserve every explicit constraint", GENERATE_SYSTEM_PROMPT)

    def test_code_request_produces_code(self):
        self.assertIn("ask for code, return the code", GENERATE_SYSTEM_PROMPT)

    def test_email_request_produces_an_email(self):
        self.assertIn("asks for an email, return the email", GENERATE_SYSTEM_PROMPT)

    def test_explanation_request_produces_an_explanation(self):
        self.assertIn(
            "ask for an explanation, return the explanation", GENERATE_SYSTEM_PROMPT
        )

    def test_structured_requests_are_covered(self):
        self.assertIn("structured data", GENERATE_SYSTEM_PROMPT)

    def test_result_is_directly_pasteable(self):
        self.assertIn("directly pasteable", GENERATE_SYSTEM_PROMPT)

    def test_does_not_ask_clarifying_questions(self):
        # The user cannot answer while push-to-talk is running.
        self.assertIn("Do not ask clarifying questions", GENERATE_SYSTEM_PROMPT)

    def test_artifact_must_be_complete_enough_to_use(self):
        self.assertIn("complete enough to use as-is", GENERATE_SYSTEM_PROMPT)


class TestPromptsDoNotLeakAcrossModes(unittest.TestCase):
    def test_refine_prompt_does_not_ask_for_an_artifact(self):
        for phrase in ("return the email", "return the code", "the requested artifact"):
            self.assertNotIn(phrase, REFINE_SYSTEM_PROMPT)

    def test_generate_prompt_does_not_talk_about_refinement(self):
        self.assertNotIn("refined text", GENERATE_SYSTEM_PROMPT)
        self.assertNotIn("REFINE", GENERATE_SYSTEM_PROMPT)

    def test_refine_prompt_does_not_talk_about_generation(self):
        self.assertNotIn("GENERATE", REFINE_SYSTEM_PROMPT)
        self.assertNotIn("the requested artifact", REFINE_SYSTEM_PROMPT)

    def test_both_prompts_state_their_output_discipline(self):
        self.assertIn("nothing else", REFINE_SYSTEM_PROMPT)
        self.assertIn("nothing else", GENERATE_SYSTEM_PROMPT)

    def test_both_prompts_forbid_a_preamble(self):
        self.assertIn("preamble", REFINE_SYSTEM_PROMPT)
        self.assertIn("meta-commentary", GENERATE_SYSTEM_PROMPT)

    def test_both_forbid_think_tags(self):
        for prompt in (REFINE_SYSTEM_PROMPT, GENERATE_SYSTEM_PROMPT):
            self.assertIn("<think>", prompt)


class TestModeContractsAreDeclared(unittest.TestCase):
    def test_every_mode_has_a_contract(self):
        self.assertEqual(set(MODE_CONTRACTS), set(Mode))

    def test_contract_fields(self):
        for contract in MODE_CONTRACTS.values():
            self.assertEqual(set(contract), {"preserve", "forbid", "output"})

    def test_refine_contract_forbids_answering(self):
        self.assertIn("answering", MODE_CONTRACTS[Mode.REFINE]["forbid"])

    def test_generate_contract_mentions_constraints(self):
        self.assertIn("constraint", MODE_CONTRACTS[Mode.GENERATE]["preserve"])


if __name__ == "__main__":
    unittest.main()
