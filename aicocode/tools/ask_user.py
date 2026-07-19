from __future__ import annotations

from pydantic import BaseModel, Field

from aicocode.tools.tool_base import Tool, ToolResult

class QuestionItem(BaseModel):
    type: str = Field(description="Question type: text, radio, select, checkbox")
    name: str = Field(description="Question identifier")
    message: str = Field(description="Question text to display")
    options: list[str] = Field(
        default_factory=list,
        description="Options for radio/select/checkbox types",
    )


class AskUserParams(BaseModel):
    questions: list[QuestionItem] = Field(
        description="List of questions to ask the user"
    )


class AskUserTool(Tool):
    name = "AskUserQuestion"
    description = (
        "Ask the user one or more questions when you need information "
        "that cannot be determined from code or context alone. Supports "
        "text input, radio (single select), select, and checkbox (multi select) "
        "question types."
    )
    params_model = AskUserParams
    category: str = "read"
    is_system_tool = True

    @staticmethod
    def format_result(params: AskUserParams, answers: dict[str, str]) -> ToolResult:
        """按 QuestionItem.name 把用户回答汇总成 ToolResult 文本。

        answers 的 key 由 UI 侧（InlineAskUserWidget）用 name 写入，
        与这里的 q.name 对齐。"""
        lines = []
        for q in params.questions:
            answer = answers.get(q.name, "(no answer)")
            lines.append(f"{q.name}: {answer}")
        return ToolResult(output="\n".join(lines))

    async def execute(self, params: AskUserParams) -> ToolResult:
        # AskUserQuestion 的弹窗交互由 Agent 通过 AskUserRequest 事件流接管
        # （见 Agent._execute_tool）：Agent 先 yield AskUserRequest 让 UI 弹窗，
        # 拿到回答后直接调 format_result 汇总，不会走到这里。保留方法仅为
        # 满足 Tool 抽象接口；直接调用时返回空答案作为安全回退。
        return AskUserTool.format_result(params, {})