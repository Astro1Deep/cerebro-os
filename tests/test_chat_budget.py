import sys
import unittest
from pathlib import Path


PAYLOAD = Path(__file__).resolve().parent.parent
if str(PAYLOAD) not in sys.path:
    sys.path.insert(0, str(PAYLOAD))


class ChatBudgetTests(unittest.TestCase):
    def test_bounded_messages_keep_the_current_question_and_drop_the_oldest_history(self):
        from cerebro_core.chat_budget import build_bounded_messages

        current_question = "Pregunta actual que no se debe perder"
        result = build_bounded_messages(
            system_prompt="Instrucciones del sistema.",
            curated_context="C" * 1_600,
            history=[
                {"role": "user", "content": "historial-antiguo-" + "A" * 2_000},
                {"role": "assistant", "content": "respuesta-reciente-" + "B" * 600},
                {"role": "user", "content": current_question},
            ],
            max_characters=3_000,
            context_characters=1_200,
        )

        self.assertLessEqual(sum(len(message["content"]) for message in result), 3_000)
        self.assertEqual(result[-1], {"role": "user", "content": current_question})
        self.assertNotIn("historial-antiguo", "\n".join(message["content"] for message in result))
        self.assertIn("respuesta-reciente", "\n".join(message["content"] for message in result))

    def test_bounded_messages_expands_when_required_exceeds_conservative_limit(self):
        from cerebro_core.chat_budget import build_bounded_messages

        system = "S" * 7_000
        user = "U" * 2_000
        # required es 9,000 caracteres, superior al presupuesto conservador de 8,000
        result = build_bounded_messages(
            system_prompt=system,
            curated_context="",
            history=[{"role": "user", "content": user}],
            max_characters=8_000,
        )
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["role"], "system")
        self.assertEqual(result[1]["role"], "user")
        self.assertEqual(result[1]["content"], user)

