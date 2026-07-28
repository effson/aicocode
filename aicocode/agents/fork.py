from __future__ import annotations

import copy

from aicocode.conversation import Conversation, Message, ToolResultBlock

FORK_BOILERPLATE_TAG = "<fork_boilerplate>"

FORK_BOILERPLATE = f"""{FORK_BOILERPLATE_TAG}
You are a forked worker process. You are NOT the main agent.
Rules (non-negotiable):
1. Do NOT fork again.
2. Do NOT converse, ask questions, or request confirmation.
3. Use tools directly: read files, search code, make changes.
4. Stay strictly within your assigned task scope.
5. Final report must be under 500 characters, starting with "Scope:".
</fork_boilerplate>"""


class ForkError(Exception):
    pass


def build_forked_messages(
    conversation: Conversation,
    task: str,
) -> Conversation:
    for msg in conversation.messages:
        if FORK_BOILERPLATE_TAG in msg.content:
            raise ForkError(
                "Cannot fork from a forked agent. "
                "Fork nesting is not allowed."
            )

    fork_conv = Conversation()
    fork_conv.messages = copy.deepcopy(conversation.messages)
    fork_conv.env_injected = conversation.env_injected
    fork_conv.long_term_injected = conversation.long_term_injected


    if fork_conv.messages:
        last = fork_conv.messages[-1]
        if last.role == "assistant" and last.tool_uses:
            existing_result_ids = set()
            if len(fork_conv.messages) >= 2:
                candidate = fork_conv.messages[-1]
                if candidate.tool_results:
                    existing_result_ids = {
                        tr.tool_use_id for tr in candidate.tool_results
                    }

            pending = [
                tu
                for tu in last.tool_uses
                if tu.tool_use_id not in existing_result_ids
            ]
            if pending:
                placeholders = [
                    ToolResultBlock(
                        tool_use_id=tu.tool_use_id,
                        content="interrupted",
                        is_error=False,
                    )
                    for tu in pending
                ]
                fork_conv.messages.append(
                    Message(
                        role="user",
                        content="",
                        tool_results=placeholders,
                    )
                )

    fork_conv.add_user_message(f"{FORK_BOILERPLATE}\n\nYour task:\n{task}")
    return fork_conv