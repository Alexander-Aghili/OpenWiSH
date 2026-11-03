"""Text transport for WSPBench's Inspect harness: render a transcript with tools, parse one tool call back."""
import importlib.util
from pathlib import Path
import unittest

from inspect_ai.model import ChatMessageAssistant, ChatMessageSystem, ChatMessageTool, ChatMessageUser
from inspect_ai.tool import ToolCall, ToolInfo, ToolParams

spec = importlib.util.spec_from_file_location("bc", Path(__file__).resolve().parents[1] / "scripts/wspbench_bc.py")
bc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bc)

TOOL = ToolInfo(name="analysis", description="Apply bounded pure algebra.", parameters=ToolParams())


class Transport(unittest.TestCase):
    def test_render_keeps_system_text_tools_and_every_turn(self):
        msgs = [ChatMessageSystem(content="RULES"), ChatMessageUser(content="TASK"),
                ChatMessageAssistant(content="", tool_calls=[ToolCall(id="c1", function="analysis", arguments={"operation": "read"})]),
                ChatMessageTool(content="RESULT", tool_call_id="c1", function="analysis")]
        instructions, prompt = bc.render(msgs, [TOOL])
        self.assertIn("RULES", instructions)
        self.assertIn("analysis", instructions)
        for part in ("TASK", '"operation": "read"', "RESULT"):
            self.assertIn(part, prompt)
        self.assertLess(prompt.index("TASK"), prompt.index("RESULT"))

    def test_render_without_tools_has_no_tool_protocol(self):
        instructions, _ = bc.render([ChatMessageSystem(content="RULES"), ChatMessageUser(content="TASK")], [])
        self.assertNotIn("tool_call", instructions)

    def test_parse_reads_one_tool_call_block(self):
        text = 'I will read.\n```tool_call\n{"function": "analysis", "arguments": {"operation": "read", "operands": ["target:w0"]}}\n```'
        content, call, error = bc.parse(text, {"analysis"})
        self.assertEqual((call.function, call.arguments["operands"]), ("analysis", ["target:w0"]))
        self.assertIsNone(error)

    def test_plain_reply_is_a_final_answer(self):
        content, call, error = bc.parse('{"program": {"op": "input", "index": 0}}', {"analysis"})
        self.assertIsNone(call)
        self.assertIsNone(error)
        self.assertIn("program", content)

    def test_usage_reconciles_like_wspbench_requires(self):
        from wspbench.budget import normalize_usage
        u = bc.usage({"usage": {"input": 10, "output": 5, "cache_read": 7, "cache_write": 2, "reasoning": 1}})
        counts = normalize_usage(u)
        self.assertEqual(counts["input_tokens_total"], 19)

    def test_a_turn_is_charged_as_one_request_when_the_context_size_is_known(self):
        from wspbench.budget import normalize_usage
        reply = {"usage": {"input": 69000, "output": 200, "cache_read": 73000, "cache_write": 0, "reasoning": 90}, "context_tokens": 66200}
        counts = normalize_usage(bc.usage(reply))
        self.assertEqual((counts["input_tokens_total"], counts["output_tokens"]), (66000, 200))

    def test_measured_transport_overhead_is_not_charged(self):
        from wspbench.budget import normalize_usage
        reply = {"usage": {"input": 69000, "output": 200, "cache_read": 73000, "cache_write": 0, "reasoning": 90}, "context_tokens": 66200}
        counts = normalize_usage(bc.usage(reply, overhead=1000))
        self.assertEqual(counts["input_tokens_total"], 65000)

    def test_malformed_or_unknown_tool_call_reports_an_error(self):
        for text in ("```tool_call\n{not json}\n```", '```tool_call\n{"function": "shell", "arguments": {}}\n```'):
            _, call, error = bc.parse(text, {"analysis"})
            self.assertIsNone(call)
            self.assertTrue(error)


if __name__ == "__main__":
    unittest.main()
