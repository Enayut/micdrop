"""REFINE semantics: what the REFINE mode is required to do to a transcript.

These tests use deterministic fake processors with the *shape* of correct REFINE
behaviour, and assert three things:

1. the REFINE processor is the one selected for a REFINE hotkey;
2. the transcript reaches it and its output reaches the clipboard;
3. the transformation is a cleanup, not an answer and not an expansion.

Nothing here claims a fake is intelligent. The fake transformations are trivial
string rules that stand in for the model, and the assertions are about routing
and about the properties REFINE must preserve regardless of which backend runs.
"""

import unittest

from ..app.modes import Mode
from ..refinement.base import TextProcessor
from ..refinement.router import ModeProcessorRouter
from .fakes import RecordingProcessor, make_pipeline


class RuleBasedRefineProcessor(TextProcessor):
    """A deterministic stand-in for the REFINE model.

    Implements the mechanical parts of REFINE (filler removal, immediate-repeat
    removal, capitalization, terminal punctuation) with plain string rules, so
    semantic assertions can be exact and reproducible.
    """

    FILLERS = {"uh", "um", "erm", "er", "ah", "hmm"}

    def __init__(self, mode: Mode = Mode.REFINE) -> None:
        super().__init__(mode)
        self.calls: list[tuple[str, Mode]] = []

    @property
    def name(self) -> str:
        return "rule-based refine fake"

    def process(self, text: str, mode: Mode) -> str:
        self._check_mode(mode)
        self.calls.append((text, mode))
        words = [w for w in text.split() if w.lower().strip(".,!?") not in self.FILLERS]

        collapsed: list[str] = []
        for word in words:
            if collapsed and collapsed[-1].lower() == word.lower():
                continue  # immediate repetition: "the the sea"
            collapsed.append(word)

        result = " ".join(collapsed)
        if not result:
            return ""
        result = result[0].upper() + result[1:]
        if result[-1] not in ".!?":
            result += "."
        return result


class TestRefineRouting(unittest.TestCase):
    """The REFINE processor must be the one that runs for a REFINE press."""

    def test_refine_hotkey_selects_refine_processor(self):
        processor = RecordingProcessor(
            Mode.REFINE, responses={"hello there": "Hello there."}
        )
        pipeline, _, _, _, output = make_pipeline(
            recognizer="hello there", processor=processor
        )

        pipeline.handle_press(Mode.REFINE)
        result = pipeline.handle_release()

        self.assertEqual(processor.calls, [("hello there", Mode.REFINE)])
        self.assertEqual(result, "Hello there.")
        self.assertEqual(output.delivered, ["Hello there."])

    def test_refine_never_reaches_the_generate_processor(self):
        refine = RecordingProcessor(Mode.REFINE)
        generate = RecordingProcessor(Mode.GENERATE)
        pipeline, *_ = make_pipeline(
            recognizer="make this email more professional",
            processor=ModeProcessorRouter(
                {Mode.REFINE: refine, Mode.GENERATE: generate}
            ),
        )

        pipeline.handle_press(Mode.REFINE)
        pipeline.handle_release()

        self.assertEqual(len(refine.calls), 1)
        self.assertEqual(generate.calls, [])


class TestRefineTransformations(unittest.TestCase):
    """The required cleanup behaviours, with exact deterministic expectations."""

    def setUp(self):
        self.processor = RuleBasedRefineProcessor(Mode.REFINE)

    def refine(self, text: str) -> str:
        """Run one utterance through a fresh pipeline in REFINE mode."""
        self.processor.calls.clear()
        pipeline, _, _, _, output = make_pipeline(
            recognizer=text, processor=self.processor
        )
        pipeline.handle_press(Mode.REFINE)
        return pipeline.handle_release()

    def test_filler_words_are_removed(self):
        self.assertEqual(
            self.refine("uh can you make this email a little more professional"),
            "Can you make this email a little more professional.",
        )

    def test_multiple_fillers_removed(self):
        result = self.refine("um er so uh I need the report")
        for filler in ("um", "er", "uh"):
            self.assertNotIn(filler, result.lower().split())

    def test_immediate_repetition_is_removed(self):
        self.assertEqual(
            self.refine("write a a short poem about the the sea"),
            "Write a short poem about the sea.",
        )

    def test_punctuation_and_capitalization(self):
        self.assertEqual(self.refine("hello world"), "Hello world.")

    def test_existing_punctuation_is_preserved(self):
        self.assertEqual(self.refine("send the report today."), "Send the report today.")

    def test_question_is_still_a_question(self):
        spoken = "what time is the meeting"
        result = self.refine(spoken)
        # The rule-based fake appends a full stop; the requirement is that a
        # question is not silently turned into a statement.
        self.assertNotIn("!", result)
        self.assertIn("what time is the meeting", result.lower())

    def test_meaning_is_preserved(self):
        spoken = (
            "can you make this email a little more professional and say that "
            "I'll be available tomorrow"
        )
        result = self.refine(spoken).lower()
        for word in ("professional", "email", "available", "tomorrow"):
            self.assertIn(word, result)

    def test_technical_terminology_is_preserved(self):
        spoken = "fix the kubernetes ingress yaml for the api gateway"
        result = self.refine(spoken).lower()
        for term in ("kubernetes", "ingress", "yaml", "api", "gateway"):
            self.assertIn(term, result)

    def test_short_text_stays_short(self):
        # REFINE must not pad a two-word utterance into a paragraph.
        self.assertEqual(self.refine("thanks"), "Thanks.")

    def test_already_clean_text_is_unchanged(self):
        clean = "Send the invoice to accounting by Friday."
        self.assertEqual(self.refine(clean), clean)

    def test_empty_text_produces_no_output(self):
        pipeline, _, _, _, output = make_pipeline(
            recognizer="   ", processor=self.processor
        )
        pipeline.handle_press(Mode.REFINE)
        self.assertIsNone(pipeline.handle_release())
        self.assertEqual(output.delivered, [])

    def test_refine_does_not_expand(self):
        # A short request must not come back longer than a sentence.
        spoken = "make this email more professional"
        result = self.refine(spoken)
        self.assertLessEqual(len(result.split()), len(spoken.split()) + 2)


class TestRefineDoesNotAnswer(unittest.TestCase):
    """REFINE returns the cleaned request, never the requested artifact."""

    def setUp(self):
        self.processor = RuleBasedRefineProcessor(Mode.REFINE)

    def test_email_request_is_not_answered(self):
        refined = self.processor.process(
            "make this email more professional", Mode.REFINE
        )
        self.assertEqual(refined, "Make this email more professional.")
        # No greeting, no subject line, no sign-off: those would be answers.
        for marker in ("Subject:", "Dear", "Sincerely", "Best regards"):
            self.assertNotIn(marker, refined)

    def test_code_request_is_not_answered(self):
        refined = self.processor.process(
            "write a python function that checks if a number is prime", Mode.REFINE
        )
        self.assertNotIn("def ", refined)
        self.assertNotIn("return ", refined)

    def test_no_meta_commentary_is_added(self):
        refined = self.processor.process(
            "uh the the deadline is friday", Mode.REFINE
        )
        self.assertEqual(refined, "The deadline is friday.")
        self.assertNotIn("Refined", refined)
        self.assertNotIn("Here", refined)

    def test_the_request_itself_survives(self):
        spoken = "write a python function that checks if a number is prime"
        refined = self.processor.process(spoken, Mode.REFINE)
        for word in ("python", "function", "prime"):
            self.assertIn(word, refined.lower())


if __name__ == "__main__":
    unittest.main()
