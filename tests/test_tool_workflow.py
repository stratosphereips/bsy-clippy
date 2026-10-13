import io
import sys
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from bsy_clippy.cli import (
    _execute_tool_command,
    _extract_tool_requests,
    _is_command_safe,
    add_tool_instructions,
    complete_turn,
)


class FakeProvider:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.inputs = []

    def batch(self, system_prompt, user_prompt, conversation_input, temperature):
        self.inputs.append(conversation_input)
        reply = next(self.replies)
        return reply, reply


class ToolWorkflowTests(unittest.TestCase):
    def test_tool_instructions_are_added_even_without_default_prompt(self):
        prompt = add_tool_instructions("")
        self.assertIn("local command", prompt)
        self.assertIn('"cmd"', prompt)

    def test_only_standalone_json_is_a_tool_request(self):
        _, requests = _extract_tool_requests('{"cmd":"ip","parms":["a"]}')
        self.assertEqual(requests, [{"cmd": "ip", "parms": ["a"]}])
        _, requests = _extract_tool_requests('Here is an example: {"cmd":"ip","parms":["a"]}')
        self.assertEqual(requests, [])

    def test_command_arguments_are_not_interpreted_by_a_shell(self):
        with redirect_stdout(io.StringIO()):
            _, output, status = _execute_tool_command(
                sys.executable, ['-c', 'import sys; print(sys.argv[1])', 'hello; echo unexpected']
            )
        self.assertEqual(output, 'hello; echo unexpected\n')
        self.assertEqual(status, 0)
        self.assertFalse(_is_command_safe('sh', ['-c', 'echo unexpected'])[0])

    def test_tool_result_is_returned_to_model_in_same_turn(self):
        provider = FakeProvider(['{"cmd":"ip","parms":["a"]}', 'Your local address is 192.0.2.10.'])
        output = io.StringIO()
        with patch('bsy_clippy.cli._execute_tool_command', return_value=('ip a', 'inet 192.0.2.10/24\n', 0)):
            with redirect_stdout(output):
                answer, history = complete_turn(provider, 'batch', 0.7, 'system', '', 'Check my IP.', io.StringIO('y\n'))
        self.assertEqual(answer, 'Your local address is 192.0.2.10.')
        self.assertIn('inet 192.0.2.10/24', provider.inputs[1])
        self.assertIn('192.0.2.10', history[0])
        self.assertIn('Your local address is 192.0.2.10.', output.getvalue())

    def test_declined_command_is_reported_without_execution(self):
        provider = FakeProvider(['{"cmd":"ip","parms":["a"]}', 'The command was declined.'])
        with patch('bsy_clippy.cli._execute_tool_command') as execute:
            with redirect_stdout(io.StringIO()):
                answer, _ = complete_turn(provider, 'batch', 0.7, 'system', '', 'Check my IP.', io.StringIO('n\n'))
        execute.assert_not_called()
        self.assertEqual(answer, 'The command was declined.')
        self.assertIn('skipped', provider.inputs[1])


if __name__ == '__main__':
    unittest.main()
