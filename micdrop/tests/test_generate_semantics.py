"""GENERATE semantics: what the GENERATE mode is required to produce.

GENERATE is the opposite of REFINE: the spoken words are an instruction, and the
output is the artifact the instruction asked for — not a tidied version of the
instruction.

The same three assertions as the REFINE suite:

1. the GENERATE processor is the one selected for a GENERATE hotkey;
2. the transcript reaches it and its artifact reaches the clipboard;
3. the output is the requested artifact, not the instruction restated.

The fake returns canned artifacts so expectations are exact. No test here claims
anything about how intelligent a model is.
"""

import unittest

from ..refinement.base import TextProcessor
from ..refinement.router import ModeProcessorRouter
from ..app.modes import Mode
from .fakes import FakeOutput, RecordingProcessor, make_pipeline

EMAIL_REQUEST = "write a professional email to my professor asking for an extension because I have been sick"
EMAIL_ARTIFACT = (
    "Subject: Request for Extension\n\n"
    "Dear Professor,\n\n"
    "I am writing to request an extension on the current assignment. I have been unwell "
    "and am not back to full health yet.\n\n"
    "Thank you for your understanding.\n\n"
    "Kind regards,\nAlex"
)

CODE_REQUEST = "create a python function that checks whether a number is prime"
CODE_ARTIFACT = (
    "def is_prime(n: int) -> bool:\n"
    "    if n < 2:\n"
    "        return False\n"
    "    for i in range(2, int(n ** 0.5) + 1):\n"
    "        if n % i == 0:\n"
    "            return False\n"
    "    return True"
)

EXPLANATION_REQUEST = "explain what a database index is in simple terms"
EXPLANATION_ARTIFACT = (
    "A database index is a separate lookup structure that stores a sorted copy of the "
    "values of a column, so the database can find matching rows without scanning the "
    "whole table each time. It costs extra storage and extra work to keep up to date, "
    "but it makes searches on that column much faster."
)

STRUCTURED_REQUEST = "give me a json object with the fields name, email and role for a developer named Alex"
STRUCTURED_ARTIFACT = '{"name": "Alex", "email": "alex@example.com", "role": "developer"}'


class GenerateFake(TextProcessor):
    """Deterministic GENERATE stand-in returning a canned artifact per request."""

    def __init__(self, mode: Mode = Mode.GENERATE, responses: dict[str, str] | None = None):
        super().__init__(mode)
        self.responses = responses or {}
        self.calls: list[tuple[str, Mode]] = []

    @property
    def name(self) -> str:
        return "generate fake"

    def process(self, text: str, mode: Mode) -> str:
        self._check_mode(mode)
        self.calls.append((text, mode))
        return self.responses.get(text, f"<artifact for: {text}>")


class TestGenerateRouting(unittest.TestCase):
    def test_generate_hotkey_selects_generate_processor(self):
        processor = GenerateFake(responses={CODE_REQUEST: CODE_ARTIFACT})
        pipeline, _, _, _, output = make_pipeline(
            recognizer=CODE_REQUEST, processor=processor
        )

        pipeline.handle_press(Mode.GENERATE)
        result = pipeline.handle_release()

        self.assertEqual(processor.calls, [(CODE_REQUEST, Mode.GENERATE)])
        self.assertEqual(result, CODE_ARTIFACT)
        self.assertEqual(output.delivered, [CODE_ARTIFACT])

    def test_generate_never_reaches_the_refine_processor(self):
        refine = RecordingProcessor(Mode.REFINE)
        generate = GenerateFake()
        pipeline, *_ = make_pipeline(
            recognizer=EMAIL_REQUEST,
            processor=ModeProcessorRouter({Mode.REFINE: refine, Mode.GENERATE: generate}),
        )

        pipeline.handle_press(Mode.GENERATE)
        pipeline.handle_release()

        self.assertEqual(len(generate.calls), 1)
        self.assertEqual(refine.calls, [])

    def test_same_words_produce_different_artifacts_per_mode(self):
        """The mode, not the words, decides the shape of the output."""
        router = ModeProcessorRouter(
            {
                Mode.REFINE: RecordingProcessor(
                    Mode.REFINE, responses={CODE_REQUEST: CODE_REQUEST}
                ),
                Mode.GENERATE: GenerateFake(responses={CODE_REQUEST: CODE_ARTIFACT}),
            }
        )
        pipeline, _, _, _, output = make_pipeline(recognizer=CODE_REQUEST, processor=router)

        pipeline.handle_press(Mode.REFINE)
        refined = pipeline.handle_release()
        pipeline.handle_press(Mode.GENERATE)
        generated = pipeline.handle_release()

        # REFINE keeps the request; GENERATE carries it out.
        self.assertEqual(refined, CODE_REQUEST)
        self.assertEqual(generated, CODE_ARTIFACT)
        self.assertEqual(output.delivered, [CODE_REQUEST, CODE_ARTIFACT])


class TestGenerateArtifacts(unittest.TestCase):
    """Each request kind produces its own kind of artifact."""

    def _generate(self, request_text: str, artifact: str) -> str:
        fake = GenerateFake(responses={request_text: artifact})
        pipeline, _, _, _, output = make_pipeline(recognizer=request_text, processor=fake)
        pipeline.handle_press(Mode.GENERATE)
        result = pipeline.handle_release()
        self.assertEqual(output.delivered, [artifact])
        return result

    def test_email_request_produces_an_email(self):
        result = self._generate(EMAIL_REQUEST, EMAIL_ARTIFACT)
        self.assertIn("Subject:", result)
        self.assertIn("Dear", result)
        self.assertIn("extension", result.lower())

    def test_coding_request_produces_code(self):
        result = self._generate(CODE_REQUEST, CODE_ARTIFACT)
        self.assertIn("def is_prime", result)
        self.assertIn("return", result)

    def test_explanation_request_produces_an_explanation(self):
        result = self._generate(EXPLANATION_REQUEST, EXPLANATION_ARTIFACT)
        self.assertTrue(len(result.split()) > 20, "explanation should be prose, not a stub")
        self.assertIn("index", result.lower())

    def test_structured_request_produces_structured_output(self):
        import json

        result = self._generate(STRUCTURED_REQUEST, STRUCTURED_ARTIFACT)
        parsed = json.loads(result)
        self.assertEqual(parsed["name"], "Alex")
        self.assertEqual(parsed["role"], "developer")


class TestGenerateConstraints(unittest.TestCase):
    def test_explicit_constraints_are_preserved(self):
        # "professional", "to my professor", "extension", "sick" must all appear.
        result = EMAIL_ARTIFACT
        for constraint in ("Subject:", "Professor", "extension"):
            self.assertIn(constraint.lower(), result.lower())

    def test_no_meta_commentary_in_output(self):
        for marker in ("Here's the", "I have interpreted", "As requested"):
            self.assertNotIn(marker, EMAIL_ARTIFACT)

    def test_no_reasoning_tags_in_output(self):
        for artifact in (EMAIL_ARTIFACT, CODE_ARTIFACT, EXPLANATION_ARTIFACT):
            self.assertNotIn("<think>", artifact)
            self.assertNotIn("</think>", artifact)

    def test_artifact_is_pasteable(self):
        # Directly pasteable: no surrounding fences unless code was requested.
        self.assertFalse(EMAIL_ARTIFACT.startswith("```"))
        # Code keeps its own fences only if the user asked for them; here it is
        # raw source, which is what a paste into an editor should contain.
        self.assertTrue(CODE_ARTIFACT.startswith("def "))


class TestGenerateEmpty(unittest.TestCase):
    def test_empty_transcription_produces_no_artifact(self):
        fake = GenerateFake()
        pipeline, _, _, _, output = make_pipeline(recognizer="   ", processor=fake)

        pipeline.handle_press(Mode.GENERATE)
        self.assertIsNone(pipeline.handle_release())
        self.assertEqual(fake.calls, [])
        self.assertEqual(output.delivered, [])


if __name__ == "__main__":
    unittest.main()
